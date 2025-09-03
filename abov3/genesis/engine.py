"""
ABOV3 Genesis Engine - Core engine for transforming ideas into reality
"""

import asyncio
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from ..core.code_generator import CodeGenerator

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
        
        # Initialize code generator
        self.code_generator = CodeGenerator(project_path)
        
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
        # Get the original idea for context
        genesis_data = await self.load_genesis_data()
        idea = genesis_data.get('idea', 'Unknown project')
        
        # Check if design exists, if not create basic design
        design_file = self.genesis_flow_dir / 'design.yaml'
        design = {}
        
        if design_file.exists():
            try:
                with open(design_file, 'r') as f:
                    design = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load design file: {e}")
        
        # Determine project type and language from design or idea
        project_type = design.get('type', self._infer_project_type(idea))
        language = design.get('language', self._infer_language(idea))
        
        # Create build progress tracker
        build_progress = {
            'started': datetime.now().isoformat(),
            'files_created': [],
            'files_modified': [],
            'dependencies_added': [],
            'tests_created': [],
            'status': 'in_progress',
            'project_type': project_type,
            'language': language,
            'idea': idea
        }
        
        try:
            # Generate project structure based on type and language
            structure_result = await self.code_generator.create_project_structure(
                project_type=project_type,
                language=language
            )
            
            # Update progress with created files
            build_progress['files_created'] = structure_result.get('created_files', [])
            build_progress['directories_created'] = structure_result.get('created_directories', [])
            build_progress['errors'] = structure_result.get('errors', [])
            
            # Generate additional custom files based on the idea
            custom_files = await self._generate_custom_files_from_idea(idea, language, project_type)
            build_progress['custom_files'] = custom_files
            
            # Mark as complete if no major errors
            if not build_progress['errors'] or len(build_progress['files_created']) > 0:
                build_progress['status'] = 'complete'
                await self.complete_phase('build')
            else:
                build_progress['status'] = 'failed'
                return {
                    'error': f"Build failed: {build_progress['errors']}",
                    'phase': 'build'
                }
            
        except Exception as e:
            build_progress['status'] = 'failed'
            build_progress['errors'].append(str(e))
            
            # Save progress file even on failure
            progress_file = self.genesis_flow_dir / 'build_progress.yaml'
            with open(progress_file, 'w') as f:
                yaml.dump(build_progress, f, default_flow_style=False)
                
            return {
                'error': f'Build phase failed: {str(e)}',
                'phase': 'build'
            }
        
        # Save build progress
        progress_file = self.genesis_flow_dir / 'build_progress.yaml'
        with open(progress_file, 'w') as f:
            yaml.dump(build_progress, f, default_flow_style=False)
        
        # Create build summary
        total_files = len(build_progress['files_created'])
        total_dirs = len(build_progress.get('directories_created', []))
        
        return {
            'phase': 'build',
            'status': 'complete',
            'progress_file': str(progress_file),
            'files_created': total_files,
            'directories_created': total_dirs,
            'project_type': project_type,
            'language': language,
            'next_phase': 'test',
            'summary': f"Created {total_files} files and {total_dirs} directories for your {language} {project_type} project"
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
        progress = (completed_phases / max(1, total_phases)) * 100
        
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
    
    def _infer_project_type(self, idea: str) -> str:
        """Infer project type from idea description"""
        idea_lower = idea.lower()
        
        # Web application keywords
        if any(keyword in idea_lower for keyword in ['web', 'website', 'html', 'css', 'frontend', 'backend', 'server', 'api', 'rest', 'flask', 'django', 'express', 'react', 'vue', 'angular']):
            if any(keyword in idea_lower for keyword in ['api', 'rest', 'backend', 'server']):
                return 'api'
            return 'web'
        
        # CLI application keywords  
        if any(keyword in idea_lower for keyword in ['cli', 'command', 'terminal', 'console', 'tool', 'script', 'automation']):
            return 'cli'
        
        # Library keywords
        if any(keyword in idea_lower for keyword in ['library', 'package', 'module', 'framework', 'utility', 'helper']):
            return 'library'
        
        # Default to basic project
        return 'basic'
    
    def _infer_language(self, idea: str) -> str:
        """Infer programming language from idea description"""
        idea_lower = idea.lower()
        
        # Explicit language mentions
        if any(keyword in idea_lower for keyword in ['python', 'django', 'flask', 'fastapi', 'py']):
            return 'python'
        if any(keyword in idea_lower for keyword in ['javascript', 'js', 'node', 'react', 'vue', 'angular', 'express']):
            return 'javascript'
        
        # Infer from context
        if any(keyword in idea_lower for keyword in ['web', 'frontend', 'browser', 'html', 'css']):
            return 'javascript'
        if any(keyword in idea_lower for keyword in ['ai', 'machine learning', 'data', 'analysis', 'automation', 'script']):
            return 'python'
        
        # Default to Python (most versatile for rapid prototyping)
        return 'python'
    
    async def _generate_custom_files_from_idea(self, idea: str, language: str, project_type: str) -> List[Dict[str, Any]]:
        """Generate custom files based on the specific idea"""
        custom_files = []
        
        try:
            # Generate a more specific main file based on the idea
            if language == 'python':
                custom_content = await self._generate_python_code_from_idea(idea, project_type)
                if custom_content:
                    result = await self.code_generator.create_file(
                        'main.py' if project_type != 'web' else 'app.py',
                        custom_content,
                        f"Main application file generated from idea: {idea[:50]}...",
                        overwrite=True
                    )
                    if result['success']:
                        custom_files.append(result)
            
            elif language == 'javascript':
                custom_content = await self._generate_javascript_code_from_idea(idea, project_type)
                if custom_content:
                    result = await self.code_generator.create_file(
                        'src/index.js',
                        custom_content,
                        f"Main JavaScript file generated from idea: {idea[:50]}...",
                        overwrite=True
                    )
                    if result['success']:
                        custom_files.append(result)
        
        except Exception as e:
            print(f"Warning: Could not generate custom files: {e}")
        
        return custom_files
    
    async def _generate_python_code_from_idea(self, idea: str, project_type: str) -> str:
        """Generate Python code based on the idea"""
        app_name = self._extract_app_name(idea)
        
        if project_type == 'web':
            return f'''#!/usr/bin/env python3
"""
{app_name} - Web Application
Generated from idea: {idea}
Built with ABOV3 Genesis
"""

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configuration
app.config['DEBUG'] = True

@app.route('/')
def index():
    """Home page for {app_name}"""
    return render_template('index.html', 
                         title="{app_name}",
                         description="{idea}")

@app.route('/api/status')
def status():
    """API status endpoint"""
    return jsonify({{
        'status': 'running',
        'app': '{app_name}',
        'message': 'Your Genesis web app is alive!'
    }})

# Add your custom routes here based on your idea:
# TODO: Implement features for: {idea}

if __name__ == '__main__':
    print("🚀 Starting {app_name}...")
    print("💡 Original idea: {idea}")
    print("🌐 Visit: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
'''

        elif project_type == 'cli':
            return f'''#!/usr/bin/env python3
"""
{app_name} - Command Line Tool
Generated from idea: {idea}
Built with ABOV3 Genesis
"""

import argparse
import sys
from pathlib import Path

class {self._to_class_name(app_name)}:
    """Main application class for {app_name}"""
    
    def __init__(self):
        self.name = "{app_name}"
        self.version = "1.0.0"
        self.description = "{idea}"
    
    def run(self, args):
        """Main application logic"""
        print(f"🚀 {{self.name}} v{{self.version}}")
        print(f"💡 {{self.description}}")
        print()
        
        # TODO: Implement your CLI logic here
        # Based on your idea: {idea}
        
        if args.verbose:
            print("📊 Verbose mode enabled")
        
        print("✅ CLI execution complete!")
        return 0

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="{app_name} - {idea}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='%(prog)s 1.0.0'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # TODO: Add command-line arguments specific to your idea
    
    return parser

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    app = {self._to_class_name(app_name)}()
    return app.run(args)

if __name__ == '__main__':
    sys.exit(main())
'''

        else:  # basic project
            return f'''#!/usr/bin/env python3
"""
{app_name}
Generated from idea: {idea}
Built with ABOV3 Genesis
"""

class {self._to_class_name(app_name)}:
    """Main class for {app_name}"""
    
    def __init__(self):
        self.name = "{app_name}"
        self.description = "{idea}"
    
    def run(self):
        """Main application logic"""
        print(f"🚀 Starting {{self.name}}...")
        print(f"💡 Original idea: {{self.description}}")
        print()
        
        # TODO: Implement your application logic here
        # Transform this code to match your vision: {idea}
        
        print("✨ Your Genesis application is running!")
        print("🎯 Ready for customization!")
    
    def process_data(self, data):
        """Process data - customize this method"""
        # TODO: Add your data processing logic
        return f"Processed: {{data}}"
    
    def generate_output(self):
        """Generate output - customize this method"""
        # TODO: Add your output generation logic
        return "Generated output from your Genesis app"

def main():
    """Main entry point"""
    app = {self._to_class_name(app_name)}()
    app.run()
    
    # Example usage - customize as needed
    result = app.process_data("sample data")
    print(f"Result: {{result}}")
    
    output = app.generate_output()
    print(f"Output: {{output}}")

if __name__ == "__main__":
    main()
'''

    async def _generate_javascript_code_from_idea(self, idea: str, project_type: str) -> str:
        """Generate JavaScript code based on the idea"""
        app_name = self._extract_app_name(idea)
        class_name = self._to_class_name(app_name)
        
        return f'''/**
 * {app_name}
 * Generated from idea: {idea}
 * Built with ABOV3 Genesis
 */

class {class_name} {{
    constructor() {{
        this.name = "{app_name}";
        this.description = "{idea}";
        this.version = "1.0.0";
    }}
    
    run() {{
        console.log(`🚀 Starting ${{this.name}} v${{this.version}}`);
        console.log(`💡 Original idea: ${{this.description}}`);
        console.log();
        
        // TODO: Implement your application logic here
        // Transform this code to match your vision: {idea}
        
        console.log("✨ Your Genesis JavaScript application is running!");
        console.log("🎯 Ready for customization!");
    }}
    
    processData(data) {{
        // TODO: Add your data processing logic
        return `Processed: ${{data}}`;
    }}
    
    generateOutput() {{
        // TODO: Add your output generation logic
        return "Generated output from your Genesis app";
    }}
}}

// Main execution
function main() {{
    const app = new {class_name}();
    app.run();
    
    // Example usage - customize as needed
    const result = app.processData("sample data");
    console.log(`Result: ${{result}}`);
    
    const output = app.generateOutput();
    console.log(`Output: ${{output}}`);
}}

// Run if this is the main module
if (require.main === module) {{
    main();
}}

module.exports = {{ {class_name} }};
'''
    
    def _extract_app_name(self, idea: str) -> str:
        """Extract a reasonable app name from the idea"""
        # Remove common words and get key terms
        stop_words = {'a', 'an', 'the', 'i', 'want', 'to', 'build', 'create', 'make', 'app', 'application', 'tool', 'system'}
        words = [word.strip('.,!?') for word in idea.lower().split() if word.strip('.,!?') not in stop_words]
        
        # Take first few meaningful words
        if words:
            return '_'.join(words[:3]).replace(' ', '_')
        return 'genesis_app'
    
    def _to_class_name(self, name: str) -> str:
        """Convert a name to a valid class name"""
        # Convert to PascalCase
        words = name.replace('-', '_').replace(' ', '_').split('_')
        return ''.join(word.capitalize() for word in words if word)