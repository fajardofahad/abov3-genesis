"""
ABOV3 Genesis Dependency Detector
Detects and manages project dependencies
"""

import asyncio
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import re
import ast

class DependencyDetector:
    """
    Dependency Detector for ABOV3 Genesis
    Analyzes project dependencies and suggests installations
    """
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.abov3_dir = self.project_path / '.abov3'
        self.deps_dir = self.abov3_dir / 'dependencies'
        
        # Ensure directories exist
        self.deps_dir.mkdir(parents=True, exist_ok=True)
        
        # Dependency files
        self.requirements_file = self.deps_dir / 'requirements.txt'
        self.installed_log = self.deps_dir / 'installed.log'
        
        # Known import mappings (import name -> package name)
        self.import_to_package = {
            # Python standard library (no installation needed)
            'os': None, 'sys': None, 'json': None, 're': None, 'pathlib': None,
            'datetime': None, 'time': None, 'collections': None, 'itertools': None,
            'functools': None, 'operator': None, 'typing': None, 'dataclasses': None,
            'enum': None, 'abc': None, 'contextlib': None, 'asyncio': None,
            'threading': None, 'multiprocessing': None, 'subprocess': None,
            'urllib': None, 'http': None, 'email': None, 'html': None, 'xml': None,
            'sqlite3': None, 'csv': None, 'configparser': None, 'logging': None,
            'unittest': None, 'pickle': None, 'copy': None, 'tempfile': None,
            'shutil': None, 'glob': None, 'fnmatch': None, 'gzip': None, 'zipfile': None,
            'tarfile': None, 'base64': None, 'hashlib': None, 'secrets': None,
            'uuid': None, 'random': None, 'math': None, 'decimal': None,
            'fractions': None, 'statistics': None, 'socket': None, 'ssl': None,
            
            # Common third-party packages
            'requests': 'requests',
            'numpy': 'numpy', 'np': 'numpy',
            'pandas': 'pandas', 'pd': 'pandas',
            'matplotlib': 'matplotlib',
            'plt': 'matplotlib',
            'seaborn': 'seaborn', 'sns': 'seaborn',
            'sklearn': 'scikit-learn',
            'scipy': 'scipy',
            'flask': 'Flask',
            'django': 'Django',
            'fastapi': 'fastapi',
            'sqlalchemy': 'SQLAlchemy',
            'pytest': 'pytest',
            'click': 'click',
            'rich': 'rich',
            'pydantic': 'pydantic',
            'jinja2': 'Jinja2',
            'yaml': 'PyYAML', 'pyyaml': 'PyYAML',
            'toml': 'toml',
            'aiohttp': 'aiohttp',
            'aiofiles': 'aiofiles',
            'psutil': 'psutil',
            'pillow': 'Pillow', 'PIL': 'Pillow',
            'cv2': 'opencv-python',
            'torch': 'torch',
            'tensorflow': 'tensorflow', 'tf': 'tensorflow',
            'transformers': 'transformers',
            'ollama': 'ollama',
            'prompt_toolkit': 'prompt_toolkit',
            'pygments': 'Pygments'
        }
        
        # Package managers and their info
        self.package_managers = {
            'pip': {
                'install_cmd': ['pip', 'install'],
                'list_cmd': ['pip', 'list', '--format=json'],
                'requirements_file': 'requirements.txt'
            },
            'conda': {
                'install_cmd': ['conda', 'install', '-y'],
                'list_cmd': ['conda', 'list', '--json'],
                'requirements_file': 'environment.yml'
            },
            'poetry': {
                'install_cmd': ['poetry', 'add'],
                'list_cmd': ['poetry', 'show', '--json'],
                'requirements_file': 'pyproject.toml'
            }
        }
        
        # Detected dependencies
        self.python_imports = set()
        self.js_imports = set()
        self.missing_packages = set()
        self.installed_packages = set()
    
    async def scan_project(self) -> Dict[str, Any]:
        """Scan the entire project for dependencies"""
        results = {
            'python': await self._scan_python_files(),
            'javascript': await self._scan_javascript_files(),
            'package_files': await self._scan_package_files(),
            'missing_dependencies': [],
            'suggestions': []
        }
        
        # Check for missing dependencies
        missing = await self._check_missing_dependencies()
        results['missing_dependencies'] = list(missing)
        
        # Generate suggestions
        results['suggestions'] = await self._generate_suggestions()
        
        return results
    
    async def _scan_python_files(self) -> Dict[str, Any]:
        """Scan Python files for imports"""
        python_files = list(self.project_path.rglob('*.py'))
        imports = set()
        
        for py_file in python_files:
            if self._should_ignore_file(py_file):
                continue
            
            try:
                file_imports = await self._extract_python_imports(py_file)
                imports.update(file_imports)
            except Exception as e:
                print(f"Error scanning {py_file}: {e}")
        
        self.python_imports = imports
        
        return {
            'files_scanned': len(python_files),
            'imports_found': list(imports),
            'unique_imports': len(imports)
        }
    
    async def _extract_python_imports(self, file_path: Path) -> Set[str]:
        """Extract imports from a Python file"""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse with AST for accuracy
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.add(name.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
            except SyntaxError:
                # Fallback to regex if AST parsing fails
                import_patterns = [
                    r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                    r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import'
                ]
                
                for line in content.split('\n'):
                    for pattern in import_patterns:
                        match = re.match(pattern, line)
                        if match:
                            imports.add(match.group(1))
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return imports
    
    async def _scan_javascript_files(self) -> Dict[str, Any]:
        """Scan JavaScript/TypeScript files for imports"""
        js_extensions = ['*.js', '*.jsx', '*.ts', '*.tsx', '*.vue']
        js_files = []
        
        for ext in js_extensions:
            js_files.extend(self.project_path.rglob(ext))
        
        imports = set()
        
        for js_file in js_files:
            if self._should_ignore_file(js_file):
                continue
            
            try:
                file_imports = await self._extract_js_imports(js_file)
                imports.update(file_imports)
            except Exception as e:
                print(f"Error scanning {js_file}: {e}")
        
        self.js_imports = imports
        
        return {
            'files_scanned': len(js_files),
            'imports_found': list(imports),
            'unique_imports': len(imports)
        }
    
    async def _extract_js_imports(self, file_path: Path) -> Set[str]:
        """Extract imports from JavaScript/TypeScript files"""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Regex patterns for different import styles
            import_patterns = [
                r"import\s+.*?from\s+['\"]([^'\"]+)['\"]",
                r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
                r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    # Extract package name (before any path separators)
                    package = match.split('/')[0]
                    if not package.startswith('.'):  # Skip relative imports
                        imports.add(package)
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return imports
    
    async def _scan_package_files(self) -> Dict[str, Any]:
        """Scan package definition files"""
        package_files = {}
        
        # Python package files
        python_files = {
            'requirements.txt': self.project_path / 'requirements.txt',
            'setup.py': self.project_path / 'setup.py',
            'pyproject.toml': self.project_path / 'pyproject.toml',
            'Pipfile': self.project_path / 'Pipfile'
        }
        
        # JavaScript package files
        js_files = {
            'package.json': self.project_path / 'package.json',
            'yarn.lock': self.project_path / 'yarn.lock',
            'package-lock.json': self.project_path / 'package-lock.json'
        }
        
        all_files = {**python_files, **js_files}
        
        for name, file_path in all_files.items():
            if file_path.exists():
                try:
                    package_files[name] = await self._parse_package_file(file_path)
                except Exception as e:
                    print(f"Error parsing {name}: {e}")
                    package_files[name] = {'error': str(e)}
        
        return package_files
    
    async def _parse_package_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a package definition file"""
        file_name = file_path.name
        
        if file_name == 'requirements.txt':
            return await self._parse_requirements_txt(file_path)
        elif file_name == 'package.json':
            return await self._parse_package_json(file_path)
        elif file_name == 'pyproject.toml':
            return await self._parse_pyproject_toml(file_path)
        elif file_name == 'setup.py':
            return await self._parse_setup_py(file_path)
        else:
            return {'type': 'unknown', 'exists': True}
    
    async def _parse_requirements_txt(self, file_path: Path) -> Dict[str, Any]:
        """Parse requirements.txt file"""
        dependencies = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    dependencies.append(line)
        
        return {
            'type': 'python_requirements',
            'dependencies': dependencies,
            'count': len(dependencies)
        }
    
    async def _parse_package_json(self, file_path: Path) -> Dict[str, Any]:
        """Parse package.json file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        dependencies = data.get('dependencies', {})
        dev_dependencies = data.get('devDependencies', {})
        
        return {
            'type': 'javascript_npm',
            'dependencies': dependencies,
            'devDependencies': dev_dependencies,
            'scripts': data.get('scripts', {}),
            'name': data.get('name'),
            'version': data.get('version')
        }
    
    async def _parse_pyproject_toml(self, file_path: Path) -> Dict[str, Any]:
        """Parse pyproject.toml file"""
        try:
            import toml
            with open(file_path, 'r') as f:
                data = toml.load(f)
            
            dependencies = []
            if 'tool' in data and 'poetry' in data['tool']:
                deps = data['tool']['poetry'].get('dependencies', {})
                dependencies = list(deps.keys())
            
            return {
                'type': 'python_poetry',
                'dependencies': dependencies,
                'data': data
            }
        except ImportError:
            return {'type': 'python_toml', 'error': 'toml package not available'}
    
    async def _parse_setup_py(self, file_path: Path) -> Dict[str, Any]:
        """Parse setup.py file (basic extraction)"""
        dependencies = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for install_requires
            requires_match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if requires_match:
                requires_str = requires_match.group(1)
                # Extract quoted strings
                deps = re.findall(r'["\']([^"\']+)["\']', requires_str)
                dependencies.extend(deps)
        
        except Exception as e:
            return {'type': 'python_setup', 'error': str(e)}
        
        return {
            'type': 'python_setup',
            'dependencies': dependencies
        }
    
    async def _check_missing_dependencies(self) -> Set[str]:
        """Check for missing Python dependencies"""
        missing = set()
        
        # Get installed packages
        installed = await self._get_installed_packages()
        self.installed_packages = installed
        
        # Check each import
        for import_name in self.python_imports:
            package_name = self.import_to_package.get(import_name, import_name)
            
            # Skip standard library imports
            if package_name is None:
                continue
            
            # Check if package is installed
            if not self._is_package_installed(package_name, installed):
                missing.add(package_name)
        
        self.missing_packages = missing
        return missing
    
    async def _get_installed_packages(self) -> Set[str]:
        """Get list of installed Python packages"""
        installed = set()
        
        try:
            # Try pip list first
            result = subprocess.run(
                ['pip', 'list', '--format=json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                installed.update(pkg['name'].lower() for pkg in packages)
        
        except Exception as e:
            print(f"Error getting installed packages: {e}")
        
        return installed
    
    def _is_package_installed(self, package_name: str, installed_packages: Set[str]) -> bool:
        """Check if a package is installed"""
        # Normalize package name
        normalized = package_name.lower().replace('-', '_').replace(' ', '_')
        
        # Check various possible names
        possible_names = [
            package_name.lower(),
            normalized,
            package_name.lower().replace('_', '-'),
            package_name.lower().replace('-', '_')
        ]
        
        return any(name in installed_packages for name in possible_names)
    
    async def _generate_suggestions(self) -> List[Dict[str, Any]]:
        """Generate dependency suggestions"""
        suggestions = []
        
        # Missing package suggestions
        for package in self.missing_packages:
            suggestions.append({
                'type': 'install_package',
                'package': package,
                'command': f'pip install {package}',
                'description': f'Install missing Python package: {package}',
                'priority': 'high'
            })
        
        # Requirements file suggestions
        if self.python_imports and not (self.project_path / 'requirements.txt').exists():
            suggestions.append({
                'type': 'create_requirements',
                'description': 'Create requirements.txt file',
                'priority': 'medium',
                'action': 'create_requirements_file'
            })
        
        # Virtual environment suggestions
        venv_indicators = ['.venv', 'venv', 'env', 'virtualenv']
        has_venv = any((self.project_path / indicator).exists() for indicator in venv_indicators)
        
        if not has_venv and self.python_imports:
            suggestions.append({
                'type': 'create_venv',
                'description': 'Create virtual environment',
                'command': 'python -m venv .venv',
                'priority': 'medium'
            })
        
        return suggestions
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if a file should be ignored"""
        ignore_patterns = [
            '.git', '.abov3', '__pycache__', 'node_modules', '.pytest_cache',
            '.venv', 'venv', 'env', 'build', 'dist', '.coverage'
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in ignore_patterns)
    
    async def install_package(self, package_name: str, package_manager: str = 'pip') -> bool:
        """Install a package"""
        if package_manager not in self.package_managers:
            print(f"Unknown package manager: {package_manager}")
            return False
        
        manager_info = self.package_managers[package_manager]
        install_cmd = manager_info['install_cmd'] + [package_name]
        
        try:
            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Log successful installation
                await self._log_installation(package_name, package_manager)
                return True
            else:
                print(f"Installation failed: {result.stderr}")
                return False
        
        except Exception as e:
            print(f"Error installing {package_name}: {e}")
            return False
    
    async def _log_installation(self, package_name: str, package_manager: str):
        """Log a successful installation"""
        log_entry = f"{datetime.now().isoformat()} - {package_manager} install {package_name}\n"
        
        with open(self.installed_log, 'a') as f:
            f.write(log_entry)
    
    async def create_requirements_file(self) -> bool:
        """Create a requirements.txt file based on detected imports"""
        requirements = []
        
        for import_name in self.python_imports:
            package_name = self.import_to_package.get(import_name)
            
            # Skip standard library imports
            if package_name is None:
                continue
            
            # Add to requirements if not already installed or if we want to pin versions
            requirements.append(package_name)
        
        # Remove duplicates and sort
        requirements = sorted(set(requirements))
        
        try:
            with open(self.requirements_file, 'w') as f:
                for req in requirements:
                    f.write(f"{req}\n")
            
            return True
        except Exception as e:
            print(f"Error creating requirements file: {e}")
            return False
    
    def get_dependency_stats(self) -> Dict[str, Any]:
        """Get dependency statistics"""
        return {
            'python_imports': len(self.python_imports),
            'js_imports': len(self.js_imports),
            'missing_packages': len(self.missing_packages),
            'installed_packages': len(self.installed_packages),
            'import_details': {
                'python': list(self.python_imports),
                'javascript': list(self.js_imports),
                'missing': list(self.missing_packages)
            }
        }
    
    async def check_virtual_environment(self) -> Dict[str, Any]:
        """Check virtual environment status"""
        import sys
        
        # Check if running in virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        # Look for virtual environment directories
        venv_dirs = []
        for venv_name in ['.venv', 'venv', 'env', 'virtualenv']:
            venv_path = self.project_path / venv_name
            if venv_path.exists():
                venv_dirs.append(str(venv_path))
        
        return {
            'in_virtual_env': in_venv,
            'python_executable': sys.executable,
            'venv_directories': venv_dirs,
            'recommendation': 'create_venv' if not in_venv and not venv_dirs else 'use_existing'
        }