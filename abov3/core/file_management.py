"""
ABOV3 Genesis - Claude Coder-Style File Management API
Advanced file operations with intelligent tracking, versioning, and AI-powered analysis
"""

import os
import shutil
import json
import hashlib
import difflib
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging
import tempfile
import zipfile
import fnmatch
from collections import defaultdict
import asyncio
import mimetypes

# Configure logging
logger = logging.getLogger(__name__)


class FileManagementAPI:
    """
    Claude Coder-style file management system with intelligent operations,
    versioning, and AI-powered analysis
    """
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.file_history = defaultdict(list)
        self.file_metadata = {}
        self.backup_dir = self.project_root / '.abov3_backups'
        self.backup_dir.mkdir(exist_ok=True)
        
        # Version control
        self.version_control = VersionControl(self.backup_dir)
        
        # File analysis
        self.file_analyzer = FileAnalyzer()
        
        # Import tracking
        self.import_tracker = ImportTracker()
        
        # File organization
        self.file_organizer = FileOrganizer()
        
        # Context awareness
        self.context_aware = True
        self.project_structure = self._analyze_project_structure()
        
        # File watching
        self.file_watchers = {}
        
        # Cache for file operations
        self.operation_cache = {}
        
    def _analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze and understand project structure"""
        structure = {
            'type': 'unknown',
            'directories': {},
            'file_patterns': {},
            'conventions': {}
        }
        
        # Detect project type
        if (self.project_root / 'package.json').exists():
            structure['type'] = 'node'
        elif (self.project_root / 'requirements.txt').exists():
            structure['type'] = 'python'
        elif (self.project_root / 'pom.xml').exists():
            structure['type'] = 'java'
        elif (self.project_root / 'Cargo.toml').exists():
            structure['type'] = 'rust'
        
        # Map common directories
        for item in self.project_root.iterdir():
            if item.is_dir():
                dir_name = item.name
                if dir_name in ['src', 'lib', 'test', 'tests', 'docs', 'config']:
                    structure['directories'][dir_name] = str(item)
        
        return structure
    
    async def create_file(self, path: Union[str, Path], content: str = "", 
                          metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create a new file with intelligent path resolution and validation
        Claude Coder style with automatic directory creation and smart naming
        """
        try:
            file_path = self._resolve_path(path)
            
            # Check if file already exists
            if file_path.exists():
                return {
                    'success': False,
                    'error': 'File already exists',
                    'suggestion': f"Use modify_file() to edit existing file or choose a different name"
                }
            
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Smart content generation based on file type
            if not content:
                content = self._generate_initial_content(file_path)
            
            # Write file
            file_path.write_text(content, encoding='utf-8')
            
            # Store metadata
            self.file_metadata[str(file_path)] = {
                'created': datetime.now().isoformat(),
                'size': len(content),
                'type': self._detect_file_type(file_path),
                'encoding': 'utf-8',
                'custom': metadata or {}
            }
            
            # Track in history
            self.file_history[str(file_path)].append({
                'action': 'created',
                'timestamp': datetime.now().isoformat(),
                'content_hash': hashlib.md5(content.encode()).hexdigest()
            })
            
            # Update import tracking if it's a code file
            if file_path.suffix in ['.py', '.js', '.ts', '.java']:
                self.import_tracker.update_file(file_path, content)
            
            logger.info(f"Created file: {file_path}")
            
            return {
                'success': True,
                'path': str(file_path),
                'absolute_path': str(file_path.absolute()),
                'size': len(content),
                'type': self._detect_file_type(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error creating file: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def modify_file(self, path: Union[str, Path], 
                         changes: Union[str, Dict, List]) -> Dict[str, Any]:
        """
        Smart file modification with diff tracking and validation
        Supports multiple change formats: direct content, diffs, or line-by-line changes
        """
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return {
                    'success': False,
                    'error': 'File does not exist',
                    'suggestion': 'Use create_file() to create a new file'
                }
            
            # Backup current version
            backup_path = self.version_control.create_backup(file_path)
            
            # Read current content
            current_content = file_path.read_text(encoding='utf-8')
            
            # Apply changes based on format
            if isinstance(changes, str):
                # Direct content replacement
                new_content = changes
            elif isinstance(changes, dict):
                # Dictionary of changes (line numbers or patterns)
                new_content = self._apply_dict_changes(current_content, changes)
            elif isinstance(changes, list):
                # List of change operations
                new_content = self._apply_list_changes(current_content, changes)
            else:
                return {
                    'success': False,
                    'error': 'Invalid change format'
                }
            
            # Validate changes
            validation = await self._validate_changes(file_path, current_content, new_content)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': 'Changes failed validation',
                    'issues': validation['issues']
                }
            
            # Generate diff
            diff = self._generate_diff(current_content, new_content)
            
            # Write new content
            file_path.write_text(new_content, encoding='utf-8')
            
            # Update history
            self.file_history[str(file_path)].append({
                'action': 'modified',
                'timestamp': datetime.now().isoformat(),
                'backup': str(backup_path),
                'diff': diff,
                'content_hash': hashlib.md5(new_content.encode()).hexdigest()
            })
            
            # Update imports if needed
            if file_path.suffix in ['.py', '.js', '.ts', '.java']:
                self.import_tracker.update_file(file_path, new_content)
            
            logger.info(f"Modified file: {file_path}")
            
            return {
                'success': True,
                'path': str(file_path),
                'backup': str(backup_path),
                'diff': diff,
                'lines_changed': len(diff.split('\n'))
            }
            
        except Exception as e:
            logger.error(f"Error modifying file: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def explain_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        AI-powered file analysis and explanation
        Provides comprehensive understanding of file purpose and structure
        """
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return {
                    'success': False,
                    'error': 'File does not exist'
                }
            
            content = file_path.read_text(encoding='utf-8')
            analysis = self.file_analyzer.analyze(file_path, content)
            
            explanation = {
                'success': True,
                'path': str(file_path),
                'type': analysis['file_type'],
                'purpose': analysis['purpose'],
                'structure': analysis['structure'],
                'dependencies': analysis.get('dependencies', []),
                'exports': analysis.get('exports', []),
                'complexity': analysis.get('complexity', {}),
                'suggestions': analysis.get('suggestions', []),
                'summary': self._generate_summary(analysis)
            }
            
            # Add language-specific analysis
            if file_path.suffix == '.py':
                explanation['python_analysis'] = self._analyze_python_file(content)
            elif file_path.suffix in ['.js', '.ts']:
                explanation['javascript_analysis'] = self._analyze_javascript_file(content)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining file: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def read_file(self, path: Union[str, Path], 
                       encoding: str = 'utf-8',
                       with_metadata: bool = True) -> Dict[str, Any]:
        """
        Enhanced file reading with metadata and analysis
        """
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return {
                    'success': False,
                    'error': 'File does not exist'
                }
            
            # Read content
            if self._is_binary_file(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read()
                result = {
                    'success': True,
                    'path': str(file_path),
                    'content': None,
                    'binary': True,
                    'size': len(content)
                }
            else:
                content = file_path.read_text(encoding=encoding)
                result = {
                    'success': True,
                    'path': str(file_path),
                    'content': content,
                    'binary': False,
                    'lines': len(content.splitlines()),
                    'size': len(content)
                }
            
            # Add metadata if requested
            if with_metadata:
                stat = file_path.stat()
                result['metadata'] = {
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'size': stat.st_size,
                    'permissions': oct(stat.st_mode),
                    'type': self._detect_file_type(file_path)
                }
                
                # Add stored metadata
                if str(file_path) in self.file_metadata:
                    result['metadata'].update(self.file_metadata[str(file_path)])
            
            return result
            
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def delete_file(self, path: Union[str, Path], 
                         backup: bool = True) -> Dict[str, Any]:
        """
        Safe file deletion with automatic backup
        """
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return {
                    'success': False,
                    'error': 'File does not exist'
                }
            
            # Create backup before deletion
            backup_path = None
            if backup:
                backup_path = self.version_control.create_backup(file_path)
            
            # Remove from import tracking
            if str(file_path) in self.import_tracker.file_imports:
                del self.import_tracker.file_imports[str(file_path)]
            
            # Delete file
            file_path.unlink()
            
            # Update history
            self.file_history[str(file_path)].append({
                'action': 'deleted',
                'timestamp': datetime.now().isoformat(),
                'backup': str(backup_path) if backup_path else None
            })
            
            logger.info(f"Deleted file: {file_path}")
            
            return {
                'success': True,
                'path': str(file_path),
                'backup': str(backup_path) if backup_path else None
            }
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def rename_file(self, old_path: Union[str, Path], 
                         new_path: Union[str, Path],
                         update_imports: bool = True) -> Dict[str, Any]:
        """
        Intelligent file renaming with automatic import updating
        """
        try:
            old_file = self._resolve_path(old_path)
            new_file = self._resolve_path(new_path)
            
            if not old_file.exists():
                return {
                    'success': False,
                    'error': 'Source file does not exist'
                }
            
            if new_file.exists():
                return {
                    'success': False,
                    'error': 'Destination file already exists'
                }
            
            # Create backup
            backup_path = self.version_control.create_backup(old_file)
            
            # Move file
            shutil.move(str(old_file), str(new_file))
            
            # Update imports if requested
            updated_files = []
            if update_imports and old_file.suffix in ['.py', '.js', '.ts']:
                updated_files = await self._update_imports_after_rename(old_file, new_file)
            
            # Update history
            self.file_history[str(new_file)] = self.file_history.get(str(old_file), [])
            self.file_history[str(new_file)].append({
                'action': 'renamed',
                'timestamp': datetime.now().isoformat(),
                'from': str(old_file),
                'to': str(new_file),
                'imports_updated': updated_files
            })
            
            # Update metadata
            if str(old_file) in self.file_metadata:
                self.file_metadata[str(new_file)] = self.file_metadata.pop(str(old_file))
            
            logger.info(f"Renamed file: {old_file} -> {new_file}")
            
            return {
                'success': True,
                'old_path': str(old_file),
                'new_path': str(new_file),
                'backup': str(backup_path),
                'imports_updated': updated_files
            }
            
        except Exception as e:
            logger.error(f"Error renaming file: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def search_files(self, pattern: str, 
                          content_search: bool = False,
                          path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Advanced file search with pattern matching and content search
        """
        try:
            search_root = self._resolve_path(path) if path else self.project_root
            results = []
            
            if content_search:
                # Search file contents
                for file_path in search_root.rglob('*'):
                    if file_path.is_file() and not self._is_binary_file(file_path):
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            if re.search(pattern, content, re.IGNORECASE):
                                matches = []
                                for i, line in enumerate(content.splitlines(), 1):
                                    if re.search(pattern, line, re.IGNORECASE):
                                        matches.append({
                                            'line': i,
                                            'content': line.strip()
                                        })
                                results.append({
                                    'path': str(file_path),
                                    'matches': matches
                                })
                        except Exception:
                            pass  # Skip files that can't be read
            else:
                # Search file names
                for file_path in search_root.rglob('*'):
                    if file_path.is_file():
                        if fnmatch.fnmatch(file_path.name, pattern):
                            results.append({
                                'path': str(file_path),
                                'name': file_path.name,
                                'size': file_path.stat().st_size,
                                'modified': datetime.fromtimestamp(
                                    file_path.stat().st_mtime
                                ).isoformat()
                            })
            
            return {
                'success': True,
                'pattern': pattern,
                'content_search': content_search,
                'results': results,
                'count': len(results)
            }
            
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_file_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive file metadata and information
        """
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return {
                    'success': False,
                    'error': 'File does not exist'
                }
            
            stat = file_path.stat()
            
            info = {
                'success': True,
                'path': str(file_path),
                'absolute_path': str(file_path.absolute()),
                'name': file_path.name,
                'extension': file_path.suffix,
                'size': stat.st_size,
                'size_human': self._human_readable_size(stat.st_size),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
                'permissions': oct(stat.st_mode),
                'is_binary': self._is_binary_file(file_path),
                'mime_type': mimetypes.guess_type(str(file_path))[0],
                'encoding': self._detect_encoding(file_path) if not self._is_binary_file(file_path) else None
            }
            
            # Add history if available
            if str(file_path) in self.file_history:
                info['history'] = self.file_history[str(file_path)][-5:]  # Last 5 actions
            
            # Add stored metadata
            if str(file_path) in self.file_metadata:
                info['custom_metadata'] = self.file_metadata[str(file_path)]
            
            # Add code-specific info
            if file_path.suffix in ['.py', '.js', '.ts', '.java']:
                content = file_path.read_text(encoding='utf-8')
                info['code_info'] = {
                    'lines': len(content.splitlines()),
                    'characters': len(content),
                    'imports': self.import_tracker.get_imports(file_path)
                }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def backup_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Create a backup of a file
        """
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return {
                    'success': False,
                    'error': 'File does not exist'
                }
            
            backup_path = self.version_control.create_backup(file_path)
            
            return {
                'success': True,
                'original': str(file_path),
                'backup': str(backup_path),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error backing up file: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def restore_file(self, path: Union[str, Path], 
                          version: Optional[Union[int, str]] = None) -> Dict[str, Any]:
        """
        Restore a file from backup
        """
        try:
            file_path = self._resolve_path(path)
            
            restored = self.version_control.restore_version(file_path, version)
            
            if restored:
                return {
                    'success': True,
                    'path': str(file_path),
                    'restored_from': str(restored),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'No backup version found'
                }
                
        except Exception as e:
            logger.error(f"Error restoring file: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path relative to project root"""
        path = Path(path)
        if not path.is_absolute():
            path = self.project_root / path
        return path
    
    def _detect_file_type(self, path: Path) -> str:
        """Detect file type from extension"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.txt': 'text'
        }
        return ext_map.get(path.suffix.lower(), 'unknown')
    
    def _is_binary_file(self, path: Path) -> bool:
        """Check if file is binary"""
        try:
            with open(path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk
        except:
            return False
    
    def _detect_encoding(self, path: Path) -> str:
        """Detect file encoding"""
        import chardet
        try:
            with open(path, 'rb') as f:
                result = chardet.detect(f.read())
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def _human_readable_size(self, size: int) -> str:
        """Convert size to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"
    
    def _generate_initial_content(self, path: Path) -> str:
        """Generate initial content based on file type"""
        templates = {
            '.py': '#!/usr/bin/env python3\n"""\nModule description\n"""\n\n',
            '.js': '// JavaScript file\n\n',
            '.ts': '// TypeScript file\n\n',
            '.html': '<!DOCTYPE html>\n<html>\n<head>\n    <title>Title</title>\n</head>\n<body>\n    \n</body>\n</html>',
            '.css': '/* CSS Stylesheet */\n\n',
            '.md': '# Title\n\n',
            '.json': '{\n    \n}',
            '.yaml': '# YAML Configuration\n\n',
            '.yml': '# YAML Configuration\n\n'
        }
        return templates.get(path.suffix, '')
    
    def _apply_dict_changes(self, content: str, changes: Dict) -> str:
        """Apply dictionary-based changes to content"""
        lines = content.splitlines()
        
        for key, value in changes.items():
            if isinstance(key, int):
                # Line number based change
                if 0 <= key < len(lines):
                    lines[key] = value
            elif isinstance(key, str):
                # Pattern-based replacement
                new_lines = []
                for line in lines:
                    new_lines.append(re.sub(key, value, line))
                lines = new_lines
        
        return '\n'.join(lines)
    
    def _apply_list_changes(self, content: str, changes: List) -> str:
        """Apply list of change operations"""
        for change in changes:
            if isinstance(change, dict):
                if change.get('action') == 'replace':
                    content = content.replace(change['find'], change['replace'])
                elif change.get('action') == 'insert':
                    lines = content.splitlines()
                    lines.insert(change['line'], change['text'])
                    content = '\n'.join(lines)
                elif change.get('action') == 'delete':
                    lines = content.splitlines()
                    if 0 <= change['line'] < len(lines):
                        del lines[change['line']]
                    content = '\n'.join(lines)
        
        return content
    
    async def _validate_changes(self, path: Path, old_content: str, new_content: str) -> Dict:
        """Validate changes for syntax and semantic correctness"""
        validation = {'valid': True, 'issues': []}
        
        # Language-specific validation
        if path.suffix == '.py':
            try:
                compile(new_content, str(path), 'exec')
            except SyntaxError as e:
                validation['valid'] = False
                validation['issues'].append(f"Python syntax error: {e}")
        elif path.suffix in ['.json']:
            try:
                json.loads(new_content)
            except json.JSONDecodeError as e:
                validation['valid'] = False
                validation['issues'].append(f"JSON syntax error: {e}")
        
        return validation
    
    def _generate_diff(self, old: str, new: str) -> str:
        """Generate unified diff between old and new content"""
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, lineterm='')
        return ''.join(diff)
    
    def _generate_summary(self, analysis: Dict) -> str:
        """Generate human-readable summary of file analysis"""
        summary_parts = []
        
        if analysis.get('purpose'):
            summary_parts.append(f"Purpose: {analysis['purpose']}")
        
        if analysis.get('complexity'):
            complexity = analysis['complexity']
            if complexity.get('cyclomatic_complexity', 0) > 10:
                summary_parts.append("High complexity - consider refactoring")
        
        if analysis.get('dependencies'):
            summary_parts.append(f"Dependencies: {len(analysis['dependencies'])} modules")
        
        return " | ".join(summary_parts) if summary_parts else "Standard file"
    
    def _analyze_python_file(self, content: str) -> Dict:
        """Analyze Python-specific features"""
        try:
            tree = ast.parse(content)
            
            analysis = {
                'classes': [],
                'functions': [],
                'imports': [],
                'globals': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis['classes'].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    analysis['functions'].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    analysis['imports'].append(node.module)
            
            return analysis
        except:
            return {}
    
    def _analyze_javascript_file(self, content: str) -> Dict:
        """Analyze JavaScript/TypeScript features"""
        analysis = {
            'functions': [],
            'classes': [],
            'imports': [],
            'exports': []
        }
        
        # Simple regex-based analysis
        function_pattern = r'function\s+(\w+)'
        class_pattern = r'class\s+(\w+)'
        import_pattern = r'import\s+.*from\s+[\'"](.+)[\'"]'
        export_pattern = r'export\s+(?:default\s+)?(\w+)'
        
        analysis['functions'] = re.findall(function_pattern, content)
        analysis['classes'] = re.findall(class_pattern, content)
        analysis['imports'] = re.findall(import_pattern, content)
        analysis['exports'] = re.findall(export_pattern, content)
        
        return analysis
    
    async def _update_imports_after_rename(self, old_path: Path, new_path: Path) -> List[str]:
        """Update imports in other files after renaming"""
        updated = []
        
        # Get relative import paths
        old_import = self._get_import_path(old_path)
        new_import = self._get_import_path(new_path)
        
        # Search for files that might import the renamed file
        for file_path in self.project_root.rglob(f'*{old_path.suffix}'):
            if file_path.is_file() and file_path != new_path:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if old_import in content:
                        new_content = content.replace(old_import, new_import)
                        file_path.write_text(new_content, encoding='utf-8')
                        updated.append(str(file_path))
                except:
                    pass  # Skip files that can't be processed
        
        return updated
    
    def _get_import_path(self, path: Path) -> str:
        """Get import path for a file"""
        try:
            relative = path.relative_to(self.project_root)
            parts = relative.parts[:-1] + (relative.stem,)
            return '.'.join(parts)
        except:
            return path.stem


class VersionControl:
    """Simple version control for file backups"""
    
    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self.versions = defaultdict(list)
    
    def create_backup(self, file_path: Path) -> Path:
        """Create a timestamped backup of a file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        self.versions[str(file_path)].append(backup_path)
        
        return backup_path
    
    def restore_version(self, file_path: Path, version: Optional[Union[int, str]] = None) -> Optional[Path]:
        """Restore a specific version of a file"""
        if str(file_path) not in self.versions:
            return None
        
        versions = self.versions[str(file_path)]
        if not versions:
            return None
        
        if version is None:
            # Restore latest backup
            backup_path = versions[-1]
        elif isinstance(version, int):
            # Restore by index
            if 0 <= version < len(versions):
                backup_path = versions[version]
            else:
                return None
        else:
            # Restore by timestamp
            for v in versions:
                if version in str(v):
                    backup_path = v
                    break
            else:
                return None
        
        shutil.copy2(backup_path, file_path)
        return backup_path


class FileAnalyzer:
    """Analyze files for structure and purpose"""
    
    def analyze(self, path: Path, content: str) -> Dict[str, Any]:
        """Analyze file content and structure"""
        analysis = {
            'file_type': self._detect_type(path),
            'purpose': self._detect_purpose(path, content),
            'structure': self._analyze_structure(content),
            'dependencies': [],
            'exports': [],
            'complexity': {},
            'suggestions': []
        }
        
        # Language-specific analysis
        if path.suffix == '.py':
            analysis.update(self._analyze_python(content))
        elif path.suffix in ['.js', '.ts']:
            analysis.update(self._analyze_javascript(content))
        
        return analysis
    
    def _detect_type(self, path: Path) -> str:
        """Detect file type"""
        return path.suffix.lstrip('.')
    
    def _detect_purpose(self, path: Path, content: str) -> str:
        """Detect file purpose from name and content"""
        name = path.stem.lower()
        
        if 'test' in name:
            return 'test file'
        elif 'config' in name:
            return 'configuration'
        elif 'main' in name or '__main__' in content:
            return 'entry point'
        elif 'util' in name or 'helper' in name:
            return 'utility functions'
        elif 'model' in name:
            return 'data model'
        elif 'view' in name or 'component' in name:
            return 'UI component'
        else:
            return 'module'
    
    def _analyze_structure(self, content: str) -> Dict:
        """Analyze general structure"""
        lines = content.splitlines()
        return {
            'lines': len(lines),
            'blank_lines': len([l for l in lines if not l.strip()]),
            'comment_lines': len([l for l in lines if l.strip().startswith(('#', '//', '/*'))])
        }
    
    def _analyze_python(self, content: str) -> Dict:
        """Python-specific analysis"""
        try:
            tree = ast.parse(content)
            
            analysis = {
                'dependencies': [],
                'exports': [],
                'complexity': {'cyclomatic': 1}
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['dependencies'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis['dependencies'].append(node.module)
                elif isinstance(node, (ast.If, ast.While, ast.For)):
                    analysis['complexity']['cyclomatic'] += 1
            
            return analysis
        except:
            return {}
    
    def _analyze_javascript(self, content: str) -> Dict:
        """JavaScript-specific analysis"""
        return {
            'dependencies': re.findall(r'import.*from\s+[\'"](.+)[\'"]', content),
            'exports': re.findall(r'export\s+(?:default\s+)?(\w+)', content)
        }


class ImportTracker:
    """Track and manage imports across files"""
    
    def __init__(self):
        self.file_imports = {}
        self.import_graph = defaultdict(set)
    
    def update_file(self, path: Path, content: str):
        """Update import tracking for a file"""
        imports = self._extract_imports(path, content)
        self.file_imports[str(path)] = imports
        
        # Update import graph
        for imp in imports:
            self.import_graph[imp].add(str(path))
    
    def get_imports(self, path: Path) -> List[str]:
        """Get imports for a file"""
        return self.file_imports.get(str(path), [])
    
    def _extract_imports(self, path: Path, content: str) -> List[str]:
        """Extract imports from file content"""
        imports = []
        
        if path.suffix == '.py':
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
            except:
                pass
        elif path.suffix in ['.js', '.ts']:
            imports = re.findall(r'import.*from\s+[\'"](.+)[\'"]', content)
        
        return imports


class FileOrganizer:
    """Organize and suggest file structure improvements"""
    
    def suggest_organization(self, project_root: Path) -> Dict[str, Any]:
        """Suggest file organization improvements"""
        suggestions = {
            'structure': [],
            'naming': [],
            'grouping': []
        }
        
        # Analyze current structure
        files_by_type = defaultdict(list)
        for file_path in project_root.rglob('*'):
            if file_path.is_file():
                files_by_type[file_path.suffix].append(file_path)
        
        # Suggest improvements
        for ext, files in files_by_type.items():
            if len(files) > 10 and ext in ['.py', '.js', '.ts']:
                suggestions['structure'].append({
                    'issue': f"Many {ext} files in root",
                    'suggestion': f"Consider organizing {ext} files into subdirectories"
                })
        
        return suggestions


# Global file manager instance
_file_manager = None

def get_file_manager(project_root: Optional[Union[str, Path]] = None) -> FileManagementAPI:
    """Get or create global file manager instance"""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManagementAPI(project_root)
    return _file_manager