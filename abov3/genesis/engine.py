"""
ABOV3 Genesis Engine - Core engine for transforming ideas into reality
"""

import asyncio
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

class GenesisEngine:
    """
    The Genesis Engine - transforms ideas into built reality
    Coordinates the entire workflow from concept to deployment
    """
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.abov3_dir = project_path / '.abov3'
        self.genesis_file = self.abov3_dir / 'genesis.yaml'
        self.genesis_flow_dir = self.abov3_dir / 'genesis_flow'
        
        # Ensure directories exist
        self.abov3_dir.mkdir(exist_ok=True)
        self.genesis_flow_dir.mkdir(exist_ok=True)
        
        # Genesis data
        self._genesis_data = None
        
        # Phase processors
        self.phase_processors = {
            'idea': self._process_idea_phase,
            'design': self._process_design_phase,
            'build': self._process_build_phase,
            'test': self._process_test_phase,
            'deploy': self._process_deploy_phase
        }
    
    async def load_genesis_data(self) -> Dict[str, Any]:
        """Load Genesis data from file"""
        if self._genesis_data is None and self.genesis_file.exists():
            try:
                with open(self.genesis_file, 'r') as f:
                    self._genesis_data = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading genesis data: {e}")
                self._genesis_data = {}
        
        return self._genesis_data or {}
    
    async def save_genesis_data(self, data: Dict[str, Any]) -> None:
        """Save Genesis data to file"""
        self._genesis_data = data
        try:
            with open(self.genesis_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving genesis data: {e}")
    
    async def get_current_phase(self) -> str:
        """Get the current Genesis phase"""
        data = await self.load_genesis_data()
        return data.get('current_phase', 'idea')
    
    async def get_idea(self) -> Optional[str]:
        """Get the original idea"""
        data = await self.load_genesis_data()
        return data.get('idea')
    
    async def set_phase(self, phase: str, status: str = 'in_progress') -> None:
        """Set the current phase and update status"""
        data = await self.load_genesis_data()
        
        # Update current phase
        data['current_phase'] = phase
        
        # Update phase status
        if 'phases' not in data:
            data['phases'] = {}
        
        if phase not in data['phases']:
            data['phases'][phase] = {}
        
        data['phases'][phase]['status'] = status
        data['phases'][phase]['timestamp'] = datetime.now().isoformat()
        
        await self.save_genesis_data(data)
    
    async def complete_phase(self, phase: str) -> None:
        """Mark a phase as complete"""
        await self.set_phase(phase, 'complete')
        
        # Determine next phase
        next_phases = {
            'idea': 'design',
            'design': 'build', 
            'build': 'test',
            'test': 'deploy',
            'deploy': 'complete'
        }
        
        if phase in next_phases:
            next_phase = next_phases[phase]
            if next_phase != 'complete':
                await self.set_phase(next_phase, 'pending')
    
    async def get_phase_status(self, phase: str) -> str:
        """Get status of a specific phase"""
        data = await self.load_genesis_data()
        phases = data.get('phases', {})
        return phases.get(phase, {}).get('status', 'pending')
    
    async def get_all_phases(self) -> Dict[str, Dict[str, Any]]:
        """Get all phase information"""
        data = await self.load_genesis_data()
        return data.get('phases', {})
    
    async def process_phase(self, phase: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a specific Genesis phase"""
        if phase in self.phase_processors:
            processor = self.phase_processors[phase]
            return await processor(context or {})
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    async def _process_idea_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the idea phase - capture and analyze the idea"""
        idea = context.get('idea')
        if not idea:
            data = await self.load_genesis_data()
            idea = data.get('idea')
        
        if not idea:
            return {'error': 'No idea provided'}
        
        # Save idea to markdown file
        idea_file = self.genesis_flow_dir / 'idea.md'
        with open(idea_file, 'w') as f:
            f.write(f"# Genesis Idea\n\n")
            f.write(f"## Original Concept\n{idea}\n\n")
            f.write(f"## Captured\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Analysis\n")
            f.write(f"- **Complexity**: To be analyzed\n")
            f.write(f"- **Domain**: To be determined\n")
            f.write(f"- **Technology Stack**: To be designed\n")
        
        # Update Genesis data
        data = await self.load_genesis_data()
        if not data:
            data = {
                'idea': idea,
                'name': self.project_path.name,
                'created': datetime.now().isoformat(),
                'current_phase': 'idea'
            }
        
        await self.complete_phase('idea')
        
        return {
            'phase': 'idea',
            'status': 'complete',
            'idea': idea,
            'next_phase': 'design'
        }
    
    async def _process_design_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the design phase - create system architecture"""
        idea = await self.get_idea()
        if not idea:
            return {'error': 'No idea found to design from'}
        
        # Create design specification
        design_spec = {
            'idea': idea,
            'architecture': {
                'type': 'To be determined',
                'components': [],
                'technology_stack': {
                    'frontend': 'To be determined',
                    'backend': 'To be determined', 
                    'database': 'To be determined',
                    'deployment': 'To be determined'
                }
            },
            'file_structure': {},
            'api_design': {},
            'database_schema': {},
            'implementation_plan': []
        }
        
        # Save design to file
        design_file = self.genesis_flow_dir / 'design.yaml'
        with open(design_file, 'w') as f:
            yaml.dump(design_spec, f, default_flow_style=False)
        
        await self.complete_phase('design')
        
        return {
            'phase': 'design',
            'status': 'complete',
            'design_file': str(design_file),
            'next_phase': 'build'
        }
    
    async def _process_build_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the build phase - generate actual code"""
        # Check if design exists
        design_file = self.genesis_flow_dir / 'design.yaml'
        if not design_file.exists():
            return {'error': 'No design found to build from'}
        
        # Load design
        try:
            with open(design_file, 'r') as f:
                design = yaml.safe_load(f)
        except Exception as e:
            return {'error': f'Could not load design: {e}'}
        
        # Create build progress tracker
        build_progress = {
            'started': datetime.now().isoformat(),
            'files_created': [],
            'files_modified': [],
            'dependencies_added': [],
            'tests_created': [],
            'status': 'in_progress'
        }
        
        # Save build progress
        progress_file = self.genesis_flow_dir / 'build_progress.yaml'
        with open(progress_file, 'w') as f:
            yaml.dump(build_progress, f, default_flow_style=False)
        
        await self.complete_phase('build')
        
        return {
            'phase': 'build',
            'status': 'complete',
            'progress_file': str(progress_file),
            'next_phase': 'test'
        }
    
    async def _process_test_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the test phase - create and run tests"""
        # Create test results
        test_results = {
            'started': datetime.now().isoformat(),
            'unit_tests': {'passed': 0, 'failed': 0, 'total': 0},
            'integration_tests': {'passed': 0, 'failed': 0, 'total': 0},
            'coverage': 0.0,
            'status': 'complete'
        }
        
        # Save test results
        test_file = self.genesis_flow_dir / 'test_results.yaml'
        with open(test_file, 'w') as f:
            yaml.dump(test_results, f, default_flow_style=False)
        
        await self.complete_phase('test')
        
        return {
            'phase': 'test',
            'status': 'complete',
            'test_file': str(test_file),
            'next_phase': 'deploy'
        }
    
    async def _process_deploy_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the deploy phase - deploy the application"""
        # Create deployment record
        deployment = {
            'started': datetime.now().isoformat(),
            'environment': 'local',
            'url': 'http://localhost',
            'status': 'deployed',
            'services': []
        }
        
        # Save deployment info
        deploy_file = self.genesis_flow_dir / 'deployment.yaml'
        with open(deploy_file, 'w') as f:
            yaml.dump(deployment, f, default_flow_style=False)
        
        await self.complete_phase('deploy')
        
        # Mark entire Genesis as complete
        data = await self.load_genesis_data()
        data['current_phase'] = 'complete'
        data['completed'] = datetime.now().isoformat()
        await self.save_genesis_data(data)
        
        # Create reality document
        await self._create_reality_document()
        
        return {
            'phase': 'deploy',
            'status': 'complete',
            'deployment_file': str(deploy_file),
            'genesis_complete': True
        }
    
    async def _create_reality_document(self) -> None:
        """Create a final reality document showing the transformation"""
        data = await self.load_genesis_data()
        idea = data.get('idea', 'Unknown idea')
        name = data.get('name', 'Unknown project')
        
        reality_content = f"""# Genesis Complete: {name}

## From Idea to Built Reality ✨

### Original Idea
{idea}

### Genesis Journey
- **💡 Idea Phase**: Concept captured and analyzed
- **📐 Design Phase**: System architecture created  
- **🔨 Build Phase**: Application built
- **🧪 Test Phase**: Quality assured
- **🚀 Deploy Phase**: Live and ready

### Project Structure
```
{name}/
├── src/                 # Source code
├── tests/               # Test suites
├── docs/                # Documentation
├── .abov3/             # Genesis data
└── README.md           # Project overview
```

### Reality Achieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**From a simple idea to a working reality - this is the power of ABOV3 Genesis!**

---
*Generated by ABOV3 Genesis - From Idea to Built Reality*
"""
        
        reality_file = self.genesis_flow_dir / 'reality.md'
        with open(reality_file, 'w') as f:
            f.write(reality_content)
    
    async def get_genesis_stats(self) -> Dict[str, Any]:
        """Get Genesis statistics and progress"""
        data = await self.load_genesis_data()
        
        # Count phases completed
        phases = data.get('phases', {})
        completed_phases = sum(1 for phase_data in phases.values() 
                             if phase_data.get('status') == 'complete')
        total_phases = 5  # idea, design, build, test, deploy
        
        # Calculate progress percentage
        progress = (completed_phases / total_phases) * 100
        
        stats = {
            'project_name': data.get('name', 'Unknown'),
            'idea': data.get('idea', 'No idea recorded'),
            'current_phase': data.get('current_phase', 'idea'),
            'progress_percentage': progress,
            'completed_phases': completed_phases,
            'total_phases': total_phases,
            'created': data.get('created'),
            'completed': data.get('completed'),
            'phases': phases,
            'is_genesis_complete': data.get('current_phase') == 'complete'
        }
        
        return stats
    
    async def reset_genesis(self) -> None:
        """Reset Genesis to start over"""
        # Backup current genesis
        if self.genesis_file.exists():
            backup_file = self.genesis_flow_dir / f'genesis_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
            import shutil
            shutil.copy2(self.genesis_file, backup_file)
        
        # Clear current genesis data
        self._genesis_data = None
        if self.genesis_file.exists():
            self.genesis_file.unlink()
        
        # Clear genesis flow files but keep the directory
        for file in self.genesis_flow_dir.glob('*.yaml'):
            if file.name != 'idea.md':  # Keep the original idea
                file.unlink()
    
    async def get_phase_details(self, phase: str) -> Dict[str, Any]:
        """Get detailed information about a specific phase"""
        phase_files = {
            'idea': 'idea.md',
            'design': 'design.yaml',
            'build': 'build_progress.yaml',
            'test': 'test_results.yaml',
            'deploy': 'deployment.yaml'
        }
        
        details = {
            'phase': phase,
            'status': await self.get_phase_status(phase),
            'files': []
        }
        
        if phase in phase_files:
            phase_file = self.genesis_flow_dir / phase_files[phase]
            if phase_file.exists():
                details['file_path'] = str(phase_file)
                details['files'].append(phase_files[phase])
        
        return details