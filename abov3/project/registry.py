"""
ABOV3 Genesis Project Registry
Manages the global registry of projects and recent access
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

class ProjectRegistry:
    """
    Global project registry for ABOV3 Genesis
    Tracks all projects, recent access, and global settings
    """
    
    def __init__(self):
        self.global_abov3_dir = Path.home() / '.abov3'
        self.registry_file = self.global_abov3_dir / 'registry.yaml'
        self.config_file = self.global_abov3_dir / 'config.yaml'
        self.stats_file = self.global_abov3_dir / 'genesis_stats.yaml'
        
        # Ensure global directory exists
        self.global_abov3_dir.mkdir(exist_ok=True)
        
        # Registry data
        self.registry_data = {}
        self.config_data = {}
        self.stats_data = {}
        
        # Load data
        self.load_registry()
        self.load_config()
        self.load_stats()
    
    def load_registry(self):
        """Load the project registry"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    self.registry_data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading registry: {e}")
                self.registry_data = {}
        
        # Initialize registry structure if empty
        if not self.registry_data:
            self.registry_data = {
                'projects': {},
                'recent': [],
                'version': '1.0.0',
                'created': datetime.now().isoformat()
            }
            self.save_registry()
    
    def load_config(self):
        """Load global configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config_data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading config: {e}")
                self.config_data = {}
        
        # Initialize config if empty
        if not self.config_data:
            self.config_data = {
                'default_agent': 'genesis-architect',
                'auto_save_interval': 30,
                'max_recent_projects': 10,
                'theme': 'genesis',
                'genz_messages': True,
                'auto_detect_language': True,
                'preferences': {
                    'show_welcome': True,
                    'check_updates': True,
                    'analytics': False
                }
            }
            self.save_config()
    
    def load_stats(self):
        """Load Genesis statistics"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    self.stats_data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading stats: {e}")
                self.stats_data = {}
        
        # Initialize stats if empty
        if not self.stats_data:
            self.stats_data = {
                'total_projects': 0,
                'genesis_projects': 0,
                'ideas_transformed': 0,
                'total_sessions': 0,
                'total_time_spent': 0,
                'lines_generated': 0,
                'files_created': 0,
                'most_used_language': 'python',
                'favorite_agent': 'genesis-architect',
                'created': datetime.now().isoformat()
            }
            self.save_stats()
    
    def save_registry(self):
        """Save the project registry"""
        try:
            with open(self.registry_file, 'w') as f:
                yaml.dump(self.registry_data, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving registry: {e}")
    
    def save_config(self):
        """Save global configuration"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def save_stats(self):
        """Save Genesis statistics"""
        try:
            with open(self.stats_file, 'w') as f:
                yaml.dump(self.stats_data, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving stats: {e}")
    
    def add_project(self, project_info: Dict[str, Any]) -> bool:
        """Add a project to the registry"""
        path = project_info.get('path')
        if not path:
            return False
        
        project_id = self._path_to_id(path)
        
        # Create project entry
        project_entry = {
            'name': project_info.get('name', Path(path).name),
            'path': str(Path(path).resolve()),
            'added': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'genesis': project_info.get('genesis', False),
            'description': project_info.get('description', ''),
            'idea': project_info.get('idea', ''),
            'language': project_info.get('language', 'unknown'),
            'framework': project_info.get('framework', 'unknown'),
            'access_count': 1
        }
        
        self.registry_data['projects'][project_id] = project_entry
        
        # Update recent list
        self._update_recent_list(project_id)
        
        # Update stats
        self.stats_data['total_projects'] += 1
        if project_info.get('genesis', False):
            self.stats_data['genesis_projects'] += 1
        
        # Save changes
        self.save_registry()
        self.save_stats()
        return True
    
    def update_project(self, path: str, updates: Dict[str, Any]) -> bool:
        """Update project information"""
        project_id = self._path_to_id(path)
        
        if project_id in self.registry_data['projects']:
            project = self.registry_data['projects'][project_id]
            project.update(updates)
            project['last_accessed'] = datetime.now().isoformat()
            
            self.save_registry()
            return True
        
        return False
    
    def get_project(self, path: str) -> Optional[Dict[str, Any]]:
        """Get project information"""
        project_id = self._path_to_id(path)
        return self.registry_data['projects'].get(project_id)
    
    def remove_project(self, path: str) -> bool:
        """Remove a project from the registry"""
        project_id = self._path_to_id(path)
        
        if project_id in self.registry_data['projects']:
            del self.registry_data['projects'][project_id]
            
            # Remove from recent list
            self.registry_data['recent'] = [
                p_id for p_id in self.registry_data['recent'] if p_id != project_id
            ]
            
            self.save_registry()
            return True
        
        return False
    
    def update_last_accessed(self, path: str):
        """Update the last accessed time for a project"""
        project_id = self._path_to_id(path)
        
        if project_id in self.registry_data['projects']:
            project = self.registry_data['projects'][project_id]
            project['last_accessed'] = datetime.now().isoformat()
            project['access_count'] = project.get('access_count', 0) + 1
            
            # Update recent list
            self._update_recent_list(project_id)
            
            # Update stats
            self.stats_data['total_sessions'] += 1
            
            self.save_registry()
            self.save_stats()
    
    def get_recent_projects(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently accessed projects"""
        recent_projects = []
        max_recent = min(limit, self.config_data.get('max_recent_projects', 10))
        
        for project_id in self.registry_data['recent'][:max_recent]:
            if project_id in self.registry_data['projects']:
                project = self.registry_data['projects'][project_id].copy()
                
                # Check if project path still exists
                if Path(project['path']).exists():
                    recent_projects.append(project)
                else:
                    # Remove non-existent project from recent list
                    self.registry_data['recent'].remove(project_id)
        
        # Save if we removed any non-existent projects
        if len(recent_projects) < len(self.registry_data['recent'][:max_recent]):
            self.save_registry()
        
        return recent_projects
    
    def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get all registered projects"""
        projects = []
        
        for project_id, project in self.registry_data['projects'].items():
            # Check if project path still exists
            if Path(project['path']).exists():
                projects.append(project.copy())
        
        # Sort by last accessed (most recent first)
        projects.sort(key=lambda p: p.get('last_accessed', ''), reverse=True)
        return projects
    
    def get_genesis_projects(self) -> List[Dict[str, Any]]:
        """Get all Genesis projects"""
        genesis_projects = []
        
        for project in self.get_all_projects():
            if project.get('genesis', False):
                genesis_projects.append(project)
        
        return genesis_projects
    
    def search_projects(self, query: str) -> List[Dict[str, Any]]:
        """Search projects by name, path, or description"""
        query = query.lower()
        matching_projects = []
        
        for project in self.get_all_projects():
            # Search in name, path, description, and idea
            searchable_text = ' '.join([
                project.get('name', ''),
                project.get('path', ''),
                project.get('description', ''),
                project.get('idea', '')
            ]).lower()
            
            if query in searchable_text:
                matching_projects.append(project)
        
        return matching_projects
    
    def cleanup_registry(self):
        """Remove projects that no longer exist"""
        projects_to_remove = []
        
        for project_id, project in self.registry_data['projects'].items():
            if not Path(project['path']).exists():
                projects_to_remove.append(project_id)
        
        # Remove non-existent projects
        for project_id in projects_to_remove:
            del self.registry_data['projects'][project_id]
            if project_id in self.registry_data['recent']:
                self.registry_data['recent'].remove(project_id)
        
        if projects_to_remove:
            self.save_registry()
        
        return len(projects_to_remove)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config_data.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """Set a configuration value"""
        self.config_data[key] = value
        self.save_config()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Genesis statistics"""
        return self.stats_data.copy()
    
    def update_stats(self, updates: Dict[str, Any]):
        """Update statistics"""
        self.stats_data.update(updates)
        self.save_stats()
    
    def increment_stat(self, stat_name: str, amount: int = 1):
        """Increment a statistic"""
        if stat_name in self.stats_data:
            self.stats_data[stat_name] += amount
            self.save_stats()
    
    def get_project_count(self) -> Dict[str, int]:
        """Get project counts"""
        all_projects = self.get_all_projects()
        return {
            'total': len(all_projects),
            'genesis': len([p for p in all_projects if p.get('genesis', False)]),
            'recent': len(self.get_recent_projects())
        }
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry data"""
        return {
            'registry': self.registry_data,
            'config': self.config_data,
            'stats': self.stats_data,
            'exported': datetime.now().isoformat()
        }
    
    def import_registry(self, data: Dict[str, Any]) -> bool:
        """Import registry data"""
        try:
            if 'registry' in data:
                self.registry_data = data['registry']
                self.save_registry()
            
            if 'config' in data:
                self.config_data.update(data['config'])
                self.save_config()
            
            if 'stats' in data:
                self.stats_data.update(data['stats'])
                self.save_stats()
            
            return True
        except Exception as e:
            print(f"Error importing registry: {e}")
            return False
    
    def _path_to_id(self, path: str) -> str:
        """Convert a path to a project ID"""
        return str(Path(path).resolve()).replace(os.sep, '_').replace(':', '_')
    
    def _update_recent_list(self, project_id: str):
        """Update the recent projects list"""
        recent = self.registry_data['recent']
        
        # Remove if already in list
        if project_id in recent:
            recent.remove(project_id)
        
        # Add to front
        recent.insert(0, project_id)
        
        # Keep only max recent
        max_recent = self.config_data.get('max_recent_projects', 10)
        self.registry_data['recent'] = recent[:max_recent]
    
    def get_project_suggestions(self, current_path: str = None) -> List[str]:
        """Get project suggestions based on common project locations"""
        suggestions = []
        
        # Common project directories
        common_dirs = [
            Path.home() / 'projects',
            Path.home() / 'Documents' / 'projects',
            Path.home() / 'code',
            Path.home() / 'dev',
            Path.home() / 'workspace',
            Path.home() / 'src'
        ]
        
        for dir_path in common_dirs:
            if dir_path.exists() and dir_path.is_dir():
                for item in dir_path.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        suggestions.append(str(item))
        
        # Remove duplicates and sort
        suggestions = list(set(suggestions))
        suggestions.sort()
        
        return suggestions[:20]  # Return top 20 suggestions