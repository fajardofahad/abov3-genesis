"""
ABOV3 Genesis - Code Generator
Handles actual file creation and code generation in project directories
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import yaml
import json

class CodeGenerator:
    """
    Code Generator for ABOV3 Genesis
    Creates and manages actual code files in the project directory
    """
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.abov3_dir = self.project_path / '.abov3'
        self.generated_files_log = self.abov3_dir / 'generated_files.yaml'
        
        # Ensure project directory exists
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.abov3_dir.mkdir(exist_ok=True)
        
        # Track generated files
        self.generated_files = self._load_generated_files_log()
    
    def _load_generated_files_log(self) -> Dict[str, Any]:
        """Load the log of generated files"""
        if self.generated_files_log.exists():
            try:
                with open(self.generated_files_log, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {'files': [], 'created': datetime.now().isoformat()}
    
    def _save_generated_files_log(self):
        """Save the log of generated files"""
        try:
            with open(self.generated_files_log, 'w') as f:
                yaml.dump(self.generated_files, f, default_flow_style=False)
        except Exception as e:
            print(f"Warning: Could not save generated files log: {e}")
    
    async def create_file(
        self, 
        file_path: Union[str, Path], 
        content: str, 
        description: str = "Generated file",
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Create a file with the given content
        
        Args:
            file_path: Path relative to project root or absolute path
            content: File content
            description: Description of what this file does
            overwrite: Whether to overwrite existing files
        
        Returns:
            Dict with creation result
        """
        # Resolve file path
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Make path relative to project if it's not absolute
        if not file_path.is_absolute():
            full_path = self.project_path / file_path
        else:
            full_path = file_path
        
        # Check if file exists
        if full_path.exists() and not overwrite:
            return {
                'success': False,
                'error': f'File {file_path} already exists. Use overwrite=True to replace it.',
                'path': str(full_path)
            }
        
        try:
            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Log the generated file
            file_info = {
                'path': str(file_path),
                'full_path': str(full_path),
                'description': description,
                'created': datetime.now().isoformat(),
                'size': len(content),
                'lines': content.count('\n') + 1 if content else 0
            }
            
            self.generated_files['files'].append(file_info)
            self._save_generated_files_log()
            
            return {
                'success': True,
                'path': str(file_path),
                'full_path': str(full_path),
                'size': len(content),
                'lines': content.count('\n') + 1 if content else 0,
                'description': description
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'path': str(file_path)
            }
    
    async def create_directory(self, dir_path: Union[str, Path]) -> Dict[str, Any]:
        """Create a directory structure"""
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)
        
        if not dir_path.is_absolute():
            full_path = self.project_path / dir_path
        else:
            full_path = dir_path
        
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            return {
                'success': True,
                'path': str(dir_path),
                'full_path': str(full_path)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'path': str(dir_path)
            }
    
    async def create_project_structure(self, project_type: str, language: str = 'python') -> Dict[str, Any]:
        """
        Create a basic project structure based on type and language
        
        Args:
            project_type: Type of project (web, cli, api, library, etc.)
            language: Programming language (python, javascript, etc.)
        
        Returns:
            Dict with creation results
        """
        results = {'created_files': [], 'created_directories': [], 'errors': []}
        
        # Define project templates
        templates = {
            'python': {
                'web': self._create_python_web_structure,
                'cli': self._create_python_cli_structure,
                'api': self._create_python_api_structure,
                'library': self._create_python_library_structure,
                'default': self._create_python_basic_structure
            },
            'javascript': {
                'web': self._create_javascript_web_structure,
                'node': self._create_javascript_node_structure,
                'react': self._create_javascript_react_structure,
                'default': self._create_javascript_basic_structure
            }
        }
        
        # Get template function
        lang_templates = templates.get(language.lower(), templates['python'])
        template_func = lang_templates.get(project_type.lower(), lang_templates['default'])
        
        # Create structure
        try:
            structure_results = await template_func()
            results.update(structure_results)
        except Exception as e:
            results['errors'].append(f"Error creating project structure: {e}")
        
        return results
    
    async def _create_python_basic_structure(self) -> Dict[str, Any]:
        """Create basic Python project structure"""
        results = {'created_files': [], 'created_directories': [], 'errors': []}
        
        # Create directories
        directories = ['src', 'tests', 'docs', 'scripts']
        for directory in directories:
            dir_result = await self.create_directory(directory)
            if dir_result['success']:
                results['created_directories'].append(dir_result)
            else:
                results['errors'].append(f"Failed to create {directory}: {dir_result['error']}")
        
        # Create files
        files_to_create = [
            {
                'path': 'main.py',
                'content': '''#!/usr/bin/env python3
"""
Main entry point for the application
"""

def main():
    """Main function"""
    print("Hello from your ABOV3 Genesis project!")
    print("This is your starting point - transform this into your vision!")

if __name__ == "__main__":
    main()
''',
                'description': 'Main entry point'
            },
            {
                'path': 'README.md',
                'content': f'''# {self.project_path.name}

Generated by ABOV3 Genesis - From Idea to Built Reality

## Description

Your project description goes here.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Development

This project was created with ABOV3 Genesis. To continue development:

1. Edit the code files in the `src/` directory
2. Add tests in the `tests/` directory
3. Update this README with your project details

## License

Your license here
''',
                'description': 'Project documentation'
            },
            {
                'path': 'requirements.txt',
                'content': '''# Project dependencies
# Add your required packages here
''',
                'description': 'Python dependencies'
            },
            {
                'path': 'src/__init__.py',
                'content': '"""Project package"""',
                'description': 'Package init file'
            },
            {
                'path': 'tests/__init__.py',
                'content': '"""Tests package"""',
                'description': 'Tests package init'
            },
            {
                'path': 'tests/test_main.py',
                'content': '''"""
Tests for main module
"""
import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class TestMain(unittest.TestCase):
    """Test cases for main functionality"""
    
    def test_placeholder(self):
        """Placeholder test"""
        self.assertTrue(True, "Your tests go here!")

if __name__ == '__main__':
    unittest.main()
''',
                'description': 'Basic test file'
            }
        ]
        
        for file_info in files_to_create:
            file_result = await self.create_file(
                file_info['path'],
                file_info['content'],
                file_info['description']
            )
            if file_result['success']:
                results['created_files'].append(file_result)
            else:
                results['errors'].append(f"Failed to create {file_info['path']}: {file_result['error']}")
        
        return results
    
    async def _create_python_web_structure(self) -> Dict[str, Any]:
        """Create Python web application structure"""
        basic_result = await self._create_python_basic_structure()
        
        # Add web-specific files
        web_files = [
            {
                'path': 'app.py',
                'content': '''"""
Web application using Flask
"""
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', 
                         title="Your ABOV3 Genesis Web App",
                         message="Welcome to your web application!")

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return {'status': 'active', 'message': 'Your API is running!'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
''',
                'description': 'Flask web application'
            },
            {
                'path': 'templates/index.html',
                'content': '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 50px auto; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 10px;
            text-align: center;
        }
        .logo {
            font-size: 3em;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">🚀</div>
        <h1>{{ title }}</h1>
        <p>{{ message }}</p>
        <p><em>Built with ABOV3 Genesis - From Idea to Built Reality</em></p>
        <div style="margin-top: 30px;">
            <a href="/api/status" style="color: #fff; text-decoration: none; background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 5px;">Check API Status</a>
        </div>
    </div>
</body>
</html>
''',
                'description': 'HTML template'
            }
        ]
        
        # Create templates directory
        dir_result = await self.create_directory('templates')
        if dir_result['success']:
            basic_result['created_directories'].append(dir_result)
        
        # Create web files
        for file_info in web_files:
            file_result = await self.create_file(
                file_info['path'],
                file_info['content'],
                file_info['description']
            )
            if file_result['success']:
                basic_result['created_files'].append(file_result)
            else:
                basic_result['errors'].append(f"Failed to create {file_info['path']}: {file_result['error']}")
        
        # Update requirements.txt
        web_requirements = '''# Web application dependencies
flask>=2.0.0
gunicorn>=20.0.0
'''
        await self.create_file('requirements.txt', web_requirements, 'Web dependencies', overwrite=True)
        
        return basic_result
    
    async def _create_python_cli_structure(self) -> Dict[str, Any]:
        """Create Python CLI application structure"""
        basic_result = await self._create_python_basic_structure()
        
        # Add CLI-specific files
        cli_files = [
            {
                'path': 'cli.py',
                'content': '''#!/usr/bin/env python3
"""
Command Line Interface for your application
"""
import argparse
import sys
from pathlib import Path

def create_parser():
    """Create command line parser"""
    parser = argparse.ArgumentParser(
        description="Your ABOV3 Genesis CLI Application",
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
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Hello command
    hello_parser = subparsers.add_parser('hello', help='Say hello')
    hello_parser.add_argument('name', nargs='?', default='World', help='Name to greet')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show application status')
    
    return parser

def cmd_hello(args):
    """Handle hello command"""
    if args.verbose:
        print(f"🚀 ABOV3 Genesis CLI says:")
    print(f"Hello, {args.name}!")
    print("✨ Your CLI application is working!")

def cmd_status(args):
    """Handle status command"""
    print("📊 Application Status:")
    print("  ✅ CLI is functional")
    print("  🎯 Ready for your customizations")
    print("  🔥 Built with ABOV3 Genesis")

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to command handlers
    if args.command == 'hello':
        cmd_hello(args)
    elif args.command == 'status':
        cmd_status(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
''',
                'description': 'CLI interface'
            }
        ]
        
        # Create CLI files
        for file_info in cli_files:
            file_result = await self.create_file(
                file_info['path'],
                file_info['content'],
                file_info['description']
            )
            if file_result['success']:
                basic_result['created_files'].append(file_result)
        
        # Update main.py to reference CLI
        cli_main_content = '''#!/usr/bin/env python3
"""
Main entry point - delegates to CLI
"""
from cli import main

if __name__ == "__main__":
    main()
'''
        await self.create_file('main.py', cli_main_content, 'Main CLI entry point', overwrite=True)
        
        return basic_result
    
    async def _create_python_api_structure(self) -> Dict[str, Any]:
        """Create Python API application structure"""
        basic_result = await self._create_python_basic_structure()
        
        # Add API-specific files
        api_files = [
            {
                'path': 'api.py',
                'content': '''#!/usr/bin/env python3
"""
REST API using FastAPI
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="Your ABOV3 Genesis API", version="1.0.0")

class StatusResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any] = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to your ABOV3 Genesis API!"}

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get API status"""
    return StatusResponse(
        status="active",
        message="Your Genesis API is running!",
        data={"version": "1.0.0", "framework": "FastAPI"}
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"healthy": True}

# Add your custom endpoints here based on your idea

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
                'description': 'FastAPI REST API application'
            }
        ]
        
        # Create API files
        for file_info in api_files:
            file_result = await self.create_file(
                file_info['path'],
                file_info['content'],
                file_info['description']
            )
            if file_result['success']:
                basic_result['created_files'].append(file_result)
        
        # Update requirements.txt for API
        api_requirements = '''# API application dependencies
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
'''
        await self.create_file('requirements.txt', api_requirements, 'API dependencies', overwrite=True)
        
        return basic_result
    
    async def _create_javascript_basic_structure(self) -> Dict[str, Any]:
        """Create basic JavaScript project structure"""
        results = {'created_files': [], 'created_directories': [], 'errors': []}
        
        # Create directories
        directories = ['src', 'tests', 'docs']
        for directory in directories:
            dir_result = await self.create_directory(directory)
            if dir_result['success']:
                results['created_directories'].append(dir_result)
        
        # Create files
        files_to_create = [
            {
                'path': 'package.json',
                'content': f'''{{\
  "name": "{self.project_path.name}",
  "version": "1.0.0",
  "description": "Generated by ABOV3 Genesis - From Idea to Built Reality",
  "main": "src/index.js",
  "scripts": {{
    "start": "node src/index.js",
    "test": "npm run test:unit",
    "test:unit": "echo \\"Add your tests here\\""
  }},
  "keywords": ["abov3", "genesis"],
  "author": "Your Name",
  "license": "MIT"
}}''',
                'description': 'Package configuration'
            },
            {
                'path': 'src/index.js',
                'content': '''/**
 * Main entry point for your ABOV3 Genesis JavaScript application
 */

function main() {
    console.log("🚀 Hello from your ABOV3 Genesis JavaScript project!");
    console.log("✨ This is your starting point - transform this into your vision!");
    
    // Your application logic goes here
    const app = new GenesisApp();
    app.run();
}

class GenesisApp {
    constructor() {
        this.name = "Your Genesis App";
        this.version = "1.0.0";
    }
    
    run() {
        console.log(`📱 Running ${this.name} v${this.version}`);
        console.log("🎯 Ready for your customizations!");
    }
}

// Run the application
if (require.main === module) {
    main();
}

module.exports = { GenesisApp };
''',
                'description': 'Main JavaScript file'
            },
            {
                'path': 'README.md',
                'content': f'''# {self.project_path.name}

Generated by ABOV3 Genesis - From Idea to Built Reality

## Description

Your JavaScript project description goes here.

## Installation

```bash
npm install
```

## Usage

```bash
npm start
```

## Development

This project was created with ABOV3 Genesis. To continue development:

1. Edit the code files in the `src/` directory
2. Add tests in the `tests/` directory
3. Update this README with your project details

## License

MIT
''',
                'description': 'Project documentation'
            }
        ]
        
        for file_info in files_to_create:
            file_result = await self.create_file(
                file_info['path'],
                file_info['content'],
                file_info['description']
            )
            if file_result['success']:
                results['created_files'].append(file_result)
        
        return results
    
    async def get_generated_files_summary(self) -> Dict[str, Any]:
        """Get summary of all generated files"""
        return {
            'total_files': len(self.generated_files.get('files', [])),
            'files': self.generated_files.get('files', []),
            'created': self.generated_files.get('created'),
            'last_updated': datetime.now().isoformat()
        }
    
    async def generate_code_from_description(
        self, 
        description: str, 
        file_path: str,
        language: str = 'python'
    ) -> Dict[str, Any]:
        """
        Generate code based on a description
        This is a placeholder for AI-generated code
        """
        # This would integrate with the AI model to generate actual code
        # For now, we'll create a template based on the description
        
        templates = {
            'python': {
                'function': '''def {name}():
    """
    {description}
    """
    # TODO: Implement {name}
    pass
''',
                'class': '''class {name}:
    """
    {description}
    """
    
    def __init__(self):
        """Initialize {name}"""
        # TODO: Add initialization logic
        pass
    
    def run(self):
        """Run {name}"""
        # TODO: Add main logic
        pass
''',
                'script': '''#!/usr/bin/env python3
"""
{description}
"""

def main():
    """Main function"""
    print("Generated from: {description}")
    # TODO: Implement your logic here

if __name__ == "__main__":
    main()
'''
            }
        }
        
        # Simple template selection logic
        description_lower = description.lower()
        if 'class' in description_lower:
            template = templates[language]['class']
            name = 'GeneratedClass'
        elif 'function' in description_lower:
            template = templates[language]['function']
            name = 'generated_function'
        else:
            template = templates[language]['script']
            name = 'generated_script'
        
        # Format template
        content = template.format(
            name=name,
            description=description
        )
        
        # Create the file
        return await self.create_file(
            file_path,
            content,
            f"Generated from description: {description[:50]}..."
        )