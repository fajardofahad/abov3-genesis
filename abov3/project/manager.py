"""
ABOV3 Genesis Project Manager
Handles individual project management, configuration, and context
"""

import asyncio
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

class ProjectManager:
    """
    Project Manager for ABOV3 Genesis
    Manages project-specific data, configuration, and context
    """
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path).resolve()
        self.abov3_dir = self.project_path / '.abov3'
        
        # Project configuration files
        self.project_config_file = self.abov3_dir / 'project.yaml'
        self.genesis_config_file = self.abov3_dir / 'genesis.yaml'
        self.context_file = self.abov3_dir / 'context' / 'project_info.yaml'
        self.file_index_file = self.abov3_dir / 'context' / 'file_index.json'
        
        # Project data
        self.project_config = {}
        self.genesis_config = {}
        self.project_context = {}
        self.file_index = {}
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            self.abov3_dir,
            self.abov3_dir / 'agents',
            self.abov3_dir / 'sessions', 
            self.abov3_dir / 'history',
            self.abov3_dir / 'genesis_flow',
            self.abov3_dir / 'tasks',
            self.abov3_dir / 'permissions',
            self.abov3_dir / 'dependencies',
            self.abov3_dir / 'context'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize the project manager"""
        await self.load_project_config()
        await self.load_genesis_config()
        await self.build_project_context()
        await self.update_file_index()
    
    async def load_project_config(self):
        """Load project configuration"""
        if self.project_config_file.exists():
            try:
                with open(self.project_config_file, 'r') as f:
                    self.project_config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading project config: {e}")
                self.project_config = {}
        else:
            # Create default project config
            self.project_config = {
                'name': self.project_path.name,
                'version': '0.1.0',
                'created': datetime.now().isoformat(),
                'genesis': False,
                'description': f'Project in {self.project_path.name}',
                'settings': {
                    'auto_save': True,
                    'auto_format': True,
                    'auto_test': False
                }
            }
            await self.save_project_config()
    
    async def load_genesis_config(self):
        """Load Genesis configuration if it exists"""
        if self.genesis_config_file.exists():
            try:
                with open(self.genesis_config_file, 'r') as f:
                    self.genesis_config = yaml.safe_load(f) or {}
                    self.project_config['genesis'] = True
            except Exception as e:
                print(f"Error loading genesis config: {e}")
                self.genesis_config = {}
    
    async def save_project_config(self):
        """Save project configuration"""
        try:
            with open(self.project_config_file, 'w') as f:
                yaml.dump(self.project_config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving project config: {e}")
    
    async def save_genesis_config(self):
        """Save Genesis configuration"""
        if self.genesis_config:
            try:
                with open(self.genesis_config_file, 'w') as f:
                    yaml.dump(self.genesis_config, f, default_flow_style=False)
            except Exception as e:
                print(f"Error saving genesis config: {e}")
    
    async def build_project_context(self):
        """Build comprehensive project context"""
        self.project_context = {
            'project': {
                'name': self.project_config.get('name', self.project_path.name),
                'path': str(self.project_path),
                'description': self.project_config.get('description', ''),
                'version': self.project_config.get('version', '0.1.0'),
                'created': self.project_config.get('created'),
                'is_genesis': bool(self.genesis_config),
                'language': await self.detect_primary_language(),
                'framework': await self.detect_framework(),
                'structure': await self.analyze_project_structure()
            },
            'genesis': self.genesis_config,
            'files': await self.get_relevant_files(),
            'dependencies': await self.analyze_dependencies(),
            'git': await self.get_git_info(),
            'updated': datetime.now().isoformat()
        }
        
        # Save context
        try:
            with open(self.context_file, 'w') as f:
                yaml.dump(self.project_context, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving project context: {e}")
    
    async def detect_primary_language(self) -> str:
        """Detect the primary programming language"""
        language_files = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx', '.ts', '.tsx'],
            'java': ['.java'],
            'cpp': ['.cpp', '.c', '.hpp', '.h'],
            'csharp': ['.cs'],
            'go': ['.go'],
            'rust': ['.rs'],
            'php': ['.php'],
            'ruby': ['.rb'],
            'kotlin': ['.kt'],
            'swift': ['.swift']
        }
        
        language_counts = {}
        
        # Count files by extension
        for file_path in self.project_path.rglob('*'):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                suffix = file_path.suffix.lower()
                for language, extensions in language_files.items():
                    if suffix in extensions:
                        language_counts[language] = language_counts.get(language, 0) + 1
                        break
        
        # Return most common language
        if language_counts:
            return max(language_counts, key=language_counts.get)
        
        return 'unknown'
    
    async def detect_framework(self) -> str:
        """Detect the framework or technology stack"""
        framework_indicators = {
            'react': ['package.json', 'src/App.js', 'src/App.jsx', 'src/App.tsx'],
            'vue': ['package.json', 'src/App.vue', 'vue.config.js'],
            'angular': ['package.json', 'angular.json', 'src/app/app.module.ts'],
            'django': ['manage.py', 'settings.py', 'requirements.txt'],
            'flask': ['app.py', 'requirements.txt'],
            'fastapi': ['main.py', 'requirements.txt'],
            'express': ['package.json', 'app.js', 'server.js'],
            'nextjs': ['next.config.js', 'package.json'],
            'spring': ['pom.xml', 'src/main/java'],
            'laravel': ['composer.json', 'artisan'],
            'rails': ['Gemfile', 'config/application.rb']
        }
        
        for framework, indicators in framework_indicators.items():
            matches = 0
            for indicator in indicators:
                if (self.project_path / indicator).exists():
                    matches += 1
            
            # If we find multiple indicators, it's likely this framework
            if matches >= 2 or (matches == 1 and len(indicators) <= 2):
                return framework
        
        # Check package.json for additional clues
        package_json = self.project_path / 'package.json'
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    dependencies = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
                    
                    if 'react' in dependencies:
                        return 'react'
                    elif 'vue' in dependencies:
                        return 'vue'
                    elif '@angular/core' in dependencies:
                        return 'angular'
                    elif 'express' in dependencies:
                        return 'express'
                    elif 'next' in dependencies:
                        return 'nextjs'
            except:
                pass
        
        return 'unknown'
    
    async def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze the project structure"""
        structure = {
            'total_files': 0,
            'total_lines': 0,
            'directories': [],
            'key_files': [],
            'file_types': {}
        }
        
        key_file_patterns = [
            'README.md', 'readme.txt', 'LICENSE', 'license.txt',
            'package.json', 'requirements.txt', 'Gemfile', 'Cargo.toml',
            'pom.xml', 'build.gradle', 'composer.json',
            'Dockerfile', 'docker-compose.yml',
            '.gitignore', '.env', '.env.example'
        ]
        
        for file_path in self.project_path.rglob('*'):
            if self._should_ignore_file(file_path):
                continue
            
            if file_path.is_file():
                structure['total_files'] += 1
                
                # Count file types
                suffix = file_path.suffix.lower() or 'no_extension'
                structure['file_types'][suffix] = structure['file_types'].get(suffix, 0) + 1
                
                # Check for key files
                if file_path.name.lower() in [f.lower() for f in key_file_patterns]:
                    structure['key_files'].append(str(file_path.relative_to(self.project_path)))
                
                # Count lines for text files
                if self._is_text_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            structure['total_lines'] += sum(1 for _ in f)
                    except:
                        pass
            
            elif file_path.is_dir() and file_path != self.abov3_dir:
                rel_path = str(file_path.relative_to(self.project_path))
                if '/' not in rel_path or rel_path.count('/') <= 2:  # Only top 2 levels
                    structure['directories'].append(rel_path)
        
        return structure
    
    async def get_relevant_files(self) -> List[Dict[str, Any]]:
        """Get list of relevant files for AI context"""
        relevant_files = []
        max_files = 50  # Limit to prevent overwhelming the AI
        
        # Priority patterns (higher priority files)
        high_priority_patterns = [
            'README*', '*.md', 'package.json', 'requirements.txt',
            'main.*', 'app.*', 'index.*', 'server.*', 'manage.py',
            '*.yaml', '*.yml', '*.json', '*.toml', '*.ini'
        ]
        
        # Get high priority files first
        high_priority_files = []
        for pattern in high_priority_patterns:
            for file_path in self.project_path.glob(pattern):
                if file_path.is_file() and not self._should_ignore_file(file_path):
                    high_priority_files.append(file_path)
        
        # Add high priority files
        for file_path in high_priority_files[:20]:  # Max 20 high priority
            file_info = await self._get_file_info(file_path)
            relevant_files.append(file_info)
        
        # Add other source files
        source_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs', '.go', '.rs', '.php', '.rb'}
        
        for file_path in self.project_path.rglob('*'):
            if (len(relevant_files) >= max_files or 
                file_path in high_priority_files or
                not file_path.is_file() or
                self._should_ignore_file(file_path)):
                continue
            
            if file_path.suffix.lower() in source_extensions:
                file_info = await self._get_file_info(file_path)
                relevant_files.append(file_info)
        
        return relevant_files[:max_files]
    
    async def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get information about a file"""
        rel_path = file_path.relative_to(self.project_path)
        
        file_info = {
            'path': str(rel_path),
            'name': file_path.name,
            'size': file_path.stat().st_size,
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'type': file_path.suffix.lower() or 'no_extension'
        }
        
        # Add line count for text files
        if self._is_text_file(file_path) and file_path.stat().st_size < 100000:  # < 100KB
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_info['lines'] = sum(1 for _ in f)
            except:
                file_info['lines'] = 0
        
        return file_info
    
    async def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies"""
        dependencies = {
            'package_managers': [],
            'packages': {},
            'dev_packages': {},
            'scripts': {}
        }
        
        # Check for different package managers
        package_files = {
            'npm': 'package.json',
            'pip': 'requirements.txt', 
            'pipenv': 'Pipfile',
            'poetry': 'pyproject.toml',
            'composer': 'composer.json',
            'maven': 'pom.xml',
            'gradle': 'build.gradle',
            'cargo': 'Cargo.toml',
            'bundler': 'Gemfile'
        }
        
        for manager, filename in package_files.items():
            file_path = self.project_path / filename
            if file_path.exists():
                dependencies['package_managers'].append(manager)
                
                # Parse specific package files
                if manager == 'npm' and filename == 'package.json':
                    await self._parse_package_json(file_path, dependencies)
                elif manager == 'pip' and filename == 'requirements.txt':
                    await self._parse_requirements_txt(file_path, dependencies)
        
        return dependencies
    
    async def _parse_package_json(self, file_path: Path, dependencies: Dict):
        """Parse package.json file"""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            dependencies['packages'] = data.get('dependencies', {})
            dependencies['dev_packages'] = data.get('devDependencies', {})
            dependencies['scripts'] = data.get('scripts', {})
        except Exception as e:
            print(f"Error parsing package.json: {e}")
    
    async def _parse_requirements_txt(self, file_path: Path, dependencies: Dict):
        """Parse requirements.txt file"""
        try:
            with open(file_path) as f:
                lines = f.readlines()
            
            packages = {}
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Simple parsing - just split on common operators
                    for op in ['>=', '<=', '==', '>', '<', '~=']:
                        if op in line:
                            package, version = line.split(op, 1)
                            packages[package.strip()] = f"{op}{version.strip()}"
                            break
                    else:
                        packages[line] = "*"
            
            dependencies['packages'] = packages
        except Exception as e:
            print(f"Error parsing requirements.txt: {e}")
    
    async def get_git_info(self) -> Dict[str, Any]:
        """Get Git repository information"""
        git_info = {
            'is_repo': False,
            'branch': None,
            'remote': None,
            'last_commit': None
        }
        
        git_dir = self.project_path / '.git'
        if git_dir.exists():
            git_info['is_repo'] = True
            
            try:
                # Get current branch
                head_file = git_dir / 'HEAD'
                if head_file.exists():
                    with open(head_file) as f:
                        head_content = f.read().strip()
                        if head_content.startswith('ref: refs/heads/'):
                            git_info['branch'] = head_content.split('/')[-1]
                
                # Get remote info
                config_file = git_dir / 'config'
                if config_file.exists():
                    with open(config_file) as f:
                        config_content = f.read()
                        if '[remote "origin"]' in config_content:
                            lines = config_content.split('\n')
                            for i, line in enumerate(lines):
                                if '[remote "origin"]' in line and i + 1 < len(lines):
                                    url_line = lines[i + 1].strip()
                                    if url_line.startswith('url ='):
                                        git_info['remote'] = url_line.split('=', 1)[1].strip()
                                    break
            except Exception as e:
                print(f"Error reading git info: {e}")
        
        return git_info
    
    async def update_file_index(self):
        """Update the file index for quick lookup"""
        file_index = {}
        
        for file_path in self.project_path.rglob('*'):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                rel_path = str(file_path.relative_to(self.project_path))
                file_index[rel_path] = {
                    'size': file_path.stat().st_size,
                    'modified': file_path.stat().st_mtime,
                    'type': file_path.suffix.lower()
                }
        
        # Save file index
        try:
            with open(self.file_index_file, 'w') as f:
                json.dump(file_index, f, indent=2)
            self.file_index = file_index
        except Exception as e:
            print(f"Error saving file index: {e}")
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if a file should be ignored"""
        ignore_patterns = [
            '.git', '.abov3', '__pycache__', 'node_modules', '.pytest_cache',
            '.venv', 'venv', 'env', '.env', 'build', 'dist', '.DS_Store',
            '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.dylib',
            '.coverage', '.nyc_output', 'coverage'
        ]
        
        path_str = str(file_path)
        for pattern in ignore_patterns:
            if pattern in path_str:
                return True
        
        return False
    
    def _is_text_file(self, file_path: Path) -> bool:
        """Check if a file is likely a text file"""
        text_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala',
            '.html', '.css', '.scss', '.less', '.xml', '.json', '.yaml', '.yml',
            '.md', '.txt', '.rst', '.toml', '.ini', '.cfg', '.conf',
            '.sql', '.sh', '.bat', '.ps1', '.dockerfile'
        }
        
        return file_path.suffix.lower() in text_extensions
    
    def get_context(self) -> Dict[str, Any]:
        """Get the current project context"""
        return self.project_context
    
    def is_genesis_project(self) -> bool:
        """Check if this is a Genesis project"""
        return bool(self.genesis_config)
    
    def get_project_name(self) -> str:
        """Get the project name"""
        return self.project_config.get('name', self.project_path.name)
    
    def get_project_description(self) -> str:
        """Get the project description"""
        return self.project_config.get('description', '')
    
    async def update_project_setting(self, key: str, value: Any):
        """Update a project setting"""
        if 'settings' not in self.project_config:
            self.project_config['settings'] = {}
        
        self.project_config['settings'][key] = value
        await self.save_project_config()
    
    async def get_project_setting(self, key: str, default: Any = None) -> Any:
        """Get a project setting"""
        return self.project_config.get('settings', {}).get(key, default)
    
    async def refresh_context(self):
        """Refresh the project context"""
        await self.build_project_context()
        await self.update_file_index()
    
    def get_abov3_dir(self) -> Path:
        """Get the .abov3 directory path"""
        return self.abov3_dir