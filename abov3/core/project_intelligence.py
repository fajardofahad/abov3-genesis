"""
ABOV3 Genesis - Project Intelligence
Analyzes project structure, learns about the codebase, and maintains project knowledge
"""

import asyncio
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter

class ProjectIntelligence:
    """
    Project Intelligence System for ABOV3 Genesis
    Learns about projects, understands codebase structure, and maintains context
    """
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.abov3_dir = self.project_path / '.abov3'
        self.intelligence_file = self.abov3_dir / 'project_intelligence.yaml'
        self.interaction_history = self.abov3_dir / 'interaction_history.yaml'
        
        # Ensure directories exist
        self.abov3_dir.mkdir(exist_ok=True)
        
        # Load existing intelligence or create new
        self.project_knowledge = self._load_project_knowledge()
        self.interactions = self._load_interaction_history()
        
        # Analysis cache
        self._file_analysis_cache = {}
        self._last_analysis_time = None
    
    def _load_project_knowledge(self) -> Dict[str, Any]:
        """Load existing project knowledge"""
        if self.intelligence_file.exists():
            try:
                with open(self.intelligence_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[DEBUG] Could not load project intelligence: {e}")
        
        return {
            'analyzed_at': None,
            'primary_language': None,
            'languages': {},
            'frameworks': [],
            'project_type': None,
            'purpose': None,
            'structure': {},
            'key_files': [],
            'dependencies': {},
            'entry_points': [],
            'patterns': {},
            'learning_notes': [],
            'confidence_score': 0.0,
            'last_updated': datetime.now().isoformat()
        }
    
    def _load_interaction_history(self) -> List[Dict[str, Any]]:
        """Load interaction history for learning"""
        if self.interaction_history.exists():
            try:
                with open(self.interaction_history, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                    return data.get('interactions', [])
            except Exception as e:
                print(f"[DEBUG] Could not load interaction history: {e}")
        
        return []
    
    def _save_project_knowledge(self):
        """Save project knowledge to disk"""
        try:
            self.project_knowledge['last_updated'] = datetime.now().isoformat()
            with open(self.intelligence_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.project_knowledge, f, default_flow_style=False)
        except Exception as e:
            print(f"[DEBUG] Could not save project intelligence: {e}")
    
    def _save_interaction_history(self):
        """Save interaction history to disk"""
        try:
            data = {
                'interactions': self.interactions,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.interaction_history, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as e:
            print(f"[DEBUG] Could not save interaction history: {e}")
    
    async def analyze_project(self, force_reanalysis: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive project analysis
        Returns updated project knowledge
        """
        print(f"[DEBUG] Starting project analysis for {self.project_path.name}")
        
        # Check if we need to reanalyze
        if not force_reanalysis and self.project_knowledge.get('analyzed_at'):
            # Check if files have changed since last analysis
            if not self._has_project_changed():
                print("[DEBUG] Project hasn't changed, using cached analysis")
                return self.project_knowledge
        
        try:
            # Step 1: Scan project structure
            structure_analysis = await self._analyze_project_structure()
            
            # Step 2: Detect languages and frameworks
            language_analysis = await self._analyze_languages_and_frameworks()
            
            # Step 3: Determine project type and purpose
            purpose_analysis = await self._analyze_project_purpose()
            
            # Step 4: Find entry points and key files
            entry_analysis = await self._analyze_entry_points()
            
            # Step 5: Analyze dependencies
            dependency_analysis = await self._analyze_dependencies()
            
            # Step 6: Look for patterns and conventions
            pattern_analysis = await self._analyze_code_patterns()
            
            # Step 7: Calculate confidence score
            confidence = self._calculate_confidence_score(
                structure_analysis, language_analysis, purpose_analysis,
                entry_analysis, dependency_analysis, pattern_analysis
            )
            
            # Update project knowledge
            self.project_knowledge.update({
                'analyzed_at': datetime.now().isoformat(),
                'structure': structure_analysis,
                'primary_language': language_analysis.get('primary'),
                'languages': language_analysis.get('languages', {}),
                'frameworks': language_analysis.get('frameworks', []),
                'project_type': purpose_analysis.get('type'),
                'purpose': purpose_analysis.get('purpose'),
                'key_files': entry_analysis.get('key_files', []),
                'entry_points': entry_analysis.get('entry_points', []),
                'dependencies': dependency_analysis,
                'patterns': pattern_analysis,
                'confidence_score': confidence
            })
            
            # Save updated knowledge
            self._save_project_knowledge()
            self._last_analysis_time = datetime.now()
            
            print(f"[DEBUG] Project analysis complete. Confidence: {confidence:.2f}")
            return self.project_knowledge
            
        except Exception as e:
            print(f"[DEBUG] Error during project analysis: {e}")
            return self.project_knowledge
    
    async def _analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze the overall project structure"""
        structure = {
            'total_files': 0,
            'directories': [],
            'file_types': Counter(),
            'size_bytes': 0,
            'depth': 0
        }
        
        try:
            for root, dirs, files in os.walk(self.project_path):
                root_path = Path(root)
                
                # Skip hidden and system directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'build', 'dist']]
                
                # Calculate depth
                relative_path = root_path.relative_to(self.project_path)
                depth = len(relative_path.parts) if str(relative_path) != '.' else 0
                structure['depth'] = max(structure['depth'], depth)
                
                # Add directory info
                if str(relative_path) != '.':
                    structure['directories'].append(str(relative_path))
                
                # Analyze files
                for file in files:
                    file_path = root_path / file
                    
                    if file_path.exists():
                        try:
                            file_size = file_path.stat().st_size
                            structure['size_bytes'] += file_size
                            structure['total_files'] += 1
                            
                            # Count file types
                            extension = file_path.suffix.lower()
                            if extension:
                                structure['file_types'][extension] += 1
                            else:
                                structure['file_types']['no_extension'] += 1
                                
                        except Exception:
                            pass
            
            return structure
            
        except Exception as e:
            print(f"[DEBUG] Error analyzing project structure: {e}")
            return structure
    
    async def _analyze_languages_and_frameworks(self) -> Dict[str, Any]:
        """Detect programming languages and frameworks"""
        analysis = {
            'languages': {},
            'frameworks': [],
            'primary': None
        }
        
        # Language detection based on file extensions
        language_extensions = {
            'python': ['.py', '.pyx', '.pyi'],
            'javascript': ['.js', '.jsx', '.mjs'],
            'typescript': ['.ts', '.tsx'],
            'html': ['.html', '.htm'],
            'css': ['.css', '.scss', '.sass', '.less'],
            'java': ['.java'],
            'cpp': ['.cpp', '.cxx', '.cc', '.c++'],
            'c': ['.c'],
            'csharp': ['.cs'],
            'go': ['.go'],
            'rust': ['.rs'],
            'php': ['.php'],
            'ruby': ['.rb'],
            'swift': ['.swift'],
            'kotlin': ['.kt'],
            'dart': ['.dart'],
            'shell': ['.sh', '.bash', '.zsh'],
            'powershell': ['.ps1'],
            'sql': ['.sql'],
            'yaml': ['.yaml', '.yml'],
            'json': ['.json'],
            'xml': ['.xml'],
            'markdown': ['.md', '.markdown']
        }
        
        try:
            # Count files by language
            file_counts = {}
            total_code_files = 0
            
            for root, dirs, files in os.walk(self.project_path):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
                
                for file in files:
                    file_path = Path(root) / file
                    extension = file_path.suffix.lower()
                    
                    for language, extensions in language_extensions.items():
                        if extension in extensions:
                            if language not in file_counts:
                                file_counts[language] = 0
                            file_counts[language] += 1
                            if language in ['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'csharp', 'go', 'rust']:
                                total_code_files += 1
                            break
            
            # Calculate percentages
            if total_code_files > 0:
                for language, count in file_counts.items():
                    if language in ['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'csharp', 'go', 'rust']:
                        percentage = (count / total_code_files) * 100
                        analysis['languages'][language] = {
                            'files': count,
                            'percentage': percentage
                        }
            
            # Determine primary language
            if analysis['languages']:
                primary = max(analysis['languages'].items(), key=lambda x: x[1]['percentage'])
                analysis['primary'] = primary[0]
            
            # Framework detection
            analysis['frameworks'] = await self._detect_frameworks()
            
            return analysis
            
        except Exception as e:
            print(f"[DEBUG] Error analyzing languages and frameworks: {e}")
            return analysis
    
    async def _detect_frameworks(self) -> List[str]:
        """Detect frameworks and libraries being used"""
        frameworks = []
        
        try:
            # Check common framework indicators
            framework_indicators = {
                # Python frameworks
                'flask': ['app.py', 'application.py', 'wsgi.py', 'requirements.txt'],
                'django': ['manage.py', 'settings.py', 'wsgi.py', 'urls.py'],
                'fastapi': ['main.py', 'app.py', 'requirements.txt'],
                'streamlit': ['streamlit_app.py', 'app.py', 'requirements.txt'],
                
                # JavaScript frameworks
                'react': ['package.json', 'src/App.js', 'src/index.js', 'public/index.html'],
                'vue': ['package.json', 'src/App.vue', 'src/main.js'],
                'angular': ['angular.json', 'src/app/', 'package.json'],
                'nextjs': ['next.config.js', 'package.json', 'pages/'],
                'express': ['package.json', 'server.js', 'app.js'],
                
                # Other frameworks
                'spring': ['pom.xml', 'src/main/java/', 'application.properties'],
                'laravel': ['composer.json', 'artisan', 'app/Http/'],
                'rails': ['Gemfile', 'config/routes.rb', 'app/controllers/']
            }
            
            for framework, indicators in framework_indicators.items():
                score = 0
                for indicator in indicators:
                    if self._file_or_dir_exists(indicator):
                        score += 1
                
                # If we find enough indicators, consider the framework present
                if score >= len(indicators) * 0.5:  # At least 50% of indicators
                    frameworks.append(framework)
            
            # Check package files for more frameworks
            await self._check_package_dependencies(frameworks)
            
            return frameworks
            
        except Exception as e:
            print(f"[DEBUG] Error detecting frameworks: {e}")
            return frameworks
    
    def _file_or_dir_exists(self, path: str) -> bool:
        """Check if a file or directory exists in the project"""
        full_path = self.project_path / path
        return full_path.exists()
    
    async def _check_package_dependencies(self, frameworks: List[str]):
        """Check package files for framework dependencies"""
        try:
            # Check package.json for Node.js frameworks
            package_json = self.project_path / 'package.json'
            if package_json.exists():
                try:
                    with open(package_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        dependencies = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
                        
                        framework_deps = {
                            'react': ['react', '@types/react'],
                            'vue': ['vue', '@vue/cli'],
                            'angular': ['@angular/core', '@angular/cli'],
                            'express': ['express'],
                            'nextjs': ['next'],
                            'svelte': ['svelte'],
                            'nuxt': ['nuxt']
                        }
                        
                        for framework, deps in framework_deps.items():
                            if any(dep in dependencies for dep in deps):
                                if framework not in frameworks:
                                    frameworks.append(framework)
                except:
                    pass
            
            # Check requirements.txt for Python frameworks
            requirements = self.project_path / 'requirements.txt'
            if requirements.exists():
                try:
                    with open(requirements, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        python_frameworks = {
                            'flask': ['flask'],
                            'django': ['django'],
                            'fastapi': ['fastapi', 'uvicorn'],
                            'streamlit': ['streamlit'],
                            'pytorch': ['torch', 'pytorch'],
                            'tensorflow': ['tensorflow', 'tf'],
                            'scikit-learn': ['sklearn', 'scikit-learn']
                        }
                        
                        for framework, deps in python_frameworks.items():
                            if any(dep in content for dep in deps):
                                if framework not in frameworks:
                                    frameworks.append(framework)
                except:
                    pass
                    
        except Exception as e:
            print(f"[DEBUG] Error checking package dependencies: {e}")
    
    async def _analyze_project_purpose(self) -> Dict[str, Any]:
        """Analyze project purpose and type"""
        analysis = {
            'type': 'unknown',
            'purpose': 'Unknown project purpose',
            'confidence': 0.0
        }
        
        try:
            # Analyze based on structure and files
            type_indicators = {
                'web_app': {
                    'files': ['index.html', 'app.py', 'server.js', 'main.js'],
                    'patterns': ['templates/', 'static/', 'public/', 'src/components/']
                },
                'api': {
                    'files': ['api.py', 'server.js', 'app.py', 'main.py'],
                    'patterns': ['routes/', 'endpoints/', 'controllers/']
                },
                'cli_tool': {
                    'files': ['cli.py', 'main.py', '__main__.py', 'bin/'],
                    'patterns': ['argparse', 'click', 'typer']
                },
                'library': {
                    'files': ['setup.py', 'pyproject.toml', 'package.json', '__init__.py'],
                    'patterns': ['lib/', 'src/', 'dist/']
                },
                'data_science': {
                    'files': ['notebook.ipynb', 'analysis.py', 'model.py'],
                    'patterns': ['.ipynb', 'data/', 'models/']
                },
                'mobile_app': {
                    'files': ['pubspec.yaml', 'android/', 'ios/'],
                    'patterns': ['lib/', 'android/', 'ios/']
                }
            }
            
            type_scores = {}
            
            for project_type, indicators in type_indicators.items():
                score = 0
                
                # Check for indicator files
                for file in indicators['files']:
                    if self._file_or_dir_exists(file):
                        score += 2
                
                # Check for patterns
                for pattern in indicators['patterns']:
                    if self._search_pattern_in_project(pattern):
                        score += 1
                
                type_scores[project_type] = score
            
            # Determine most likely type
            if type_scores:
                best_type = max(type_scores.items(), key=lambda x: x[1])
                if best_type[1] > 0:
                    analysis['type'] = best_type[0]
                    analysis['confidence'] = min(best_type[1] / 5.0, 1.0)  # Normalize to 0-1
            
            # Generate purpose description
            analysis['purpose'] = await self._generate_purpose_description(analysis['type'])
            
            return analysis
            
        except Exception as e:
            print(f"[DEBUG] Error analyzing project purpose: {e}")
            return analysis
    
    def _search_pattern_in_project(self, pattern: str) -> bool:
        """Search for a pattern in project files or directories"""
        try:
            # Check if it's a directory pattern
            if pattern.endswith('/'):
                for root, dirs, files in os.walk(self.project_path):
                    if pattern[:-1] in dirs:
                        return True
            
            # Check if it's a file extension pattern
            if pattern.startswith('.'):
                for root, dirs, files in os.walk(self.project_path):
                    for file in files:
                        if file.endswith(pattern):
                            return True
            
            # Check if it's a general pattern in file contents or names
            for root, dirs, files in os.walk(self.project_path):
                for file in files:
                    if pattern in file.lower():
                        return True
            
            return False
            
        except Exception:
            return False
    
    async def _generate_purpose_description(self, project_type: str) -> str:
        """Generate a purpose description based on project type"""
        descriptions = {
            'web_app': 'A web application that serves content through a browser interface',
            'api': 'A REST API or web service that provides data endpoints',
            'cli_tool': 'A command-line tool or script for terminal-based operations',
            'library': 'A reusable library or package for other developers',
            'data_science': 'A data analysis or machine learning project',
            'mobile_app': 'A mobile application for smartphones or tablets',
            'unknown': 'A software project with unclear or mixed purposes'
        }
        
        return descriptions.get(project_type, descriptions['unknown'])
    
    async def _analyze_entry_points(self) -> Dict[str, Any]:
        """Find main entry points and key files"""
        analysis = {
            'entry_points': [],
            'key_files': []
        }
        
        try:
            # Common entry point patterns
            entry_patterns = [
                'main.py', 'app.py', 'server.py', 'run.py', '__main__.py',
                'index.js', 'server.js', 'app.js', 'main.js',
                'index.html', 'main.html',
                'Main.java', 'Application.java',
                'main.cpp', 'main.c',
                'Program.cs', 'Main.cs'
            ]
            
            # Find entry points
            for pattern in entry_patterns:
                if self._file_or_dir_exists(pattern):
                    analysis['entry_points'].append(pattern)
            
            # Find other key files
            key_patterns = [
                'README.md', 'requirements.txt', 'package.json', 'Dockerfile',
                'setup.py', 'pyproject.toml', 'Cargo.toml', 'pom.xml',
                'Gemfile', 'composer.json', 'go.mod'
            ]
            
            for pattern in key_patterns:
                if self._file_or_dir_exists(pattern):
                    analysis['key_files'].append(pattern)
            
            return analysis
            
        except Exception as e:
            print(f"[DEBUG] Error analyzing entry points: {e}")
            return analysis
    
    async def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies"""
        dependencies = {
            'package_files': [],
            'external_libs': [],
            'internal_imports': []
        }
        
        try:
            # Check for package files
            package_files = [
                'requirements.txt', 'package.json', 'Pipfile', 'poetry.lock',
                'Gemfile', 'composer.json', 'pom.xml', 'Cargo.toml'
            ]
            
            for file in package_files:
                if self._file_or_dir_exists(file):
                    dependencies['package_files'].append(file)
            
            return dependencies
            
        except Exception as e:
            print(f"[DEBUG] Error analyzing dependencies: {e}")
            return dependencies
    
    async def _analyze_code_patterns(self) -> Dict[str, Any]:
        """Analyze coding patterns and conventions"""
        patterns = {
            'naming_convention': 'unknown',
            'indentation': 'unknown',
            'common_patterns': []
        }
        
        try:
            # This would involve more complex analysis
            # For now, return basic structure
            return patterns
            
        except Exception as e:
            print(f"[DEBUG] Error analyzing code patterns: {e}")
            return patterns
    
    def _calculate_confidence_score(self, *analyses) -> float:
        """Calculate overall confidence score for the analysis"""
        try:
            total_confidence = 0.0
            valid_analyses = 0
            
            for analysis in analyses:
                if isinstance(analysis, dict):
                    if 'confidence' in analysis:
                        total_confidence += analysis['confidence']
                        valid_analyses += 1
                    else:
                        # Assign basic confidence based on data richness
                        if analysis:
                            total_confidence += 0.5
                            valid_analyses += 1
            
            return total_confidence / max(valid_analyses, 1)
            
        except Exception:
            return 0.0
    
    def _has_project_changed(self) -> bool:
        """Check if project has changed since last analysis"""
        try:
            if not self._last_analysis_time:
                return True
            
            # Check modification times of key files
            for root, dirs, files in os.walk(self.project_path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.suffix in ['.py', '.js', '.html', '.css', '.java', '.cpp']:
                        try:
                            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if mtime > self._last_analysis_time:
                                return True
                        except:
                            continue
            
            return False
            
        except Exception:
            return True
    
    async def record_interaction(self, user_request: str, ai_response: str, action_taken: str = None):
        """Record user interaction for learning"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_request': user_request,
            'ai_response_summary': ai_response[:200] + '...' if len(ai_response) > 200 else ai_response,
            'action_taken': action_taken,
            'project_state': {
                'primary_language': self.project_knowledge.get('primary_language'),
                'project_type': self.project_knowledge.get('project_type'),
                'files_modified': []  # This would be populated by the calling code
            }
        }
        
        self.interactions.append(interaction)
        
        # Keep only last 100 interactions
        if len(self.interactions) > 100:
            self.interactions = self.interactions[-100:]
        
        self._save_interaction_history()
    
    def get_project_summary(self) -> str:
        """Get a human-readable project summary"""
        knowledge = self.project_knowledge
        
        if knowledge.get('confidence_score', 0) < 0.3:
            return "Project analysis is still in progress. Limited information available."
        
        summary_parts = []
        
        # Basic info
        if knowledge.get('primary_language'):
            summary_parts.append(f"Primary language: {knowledge['primary_language'].title()}")
        
        if knowledge.get('project_type'):
            summary_parts.append(f"Project type: {knowledge['project_type'].replace('_', ' ').title()}")
        
        if knowledge.get('purpose'):
            summary_parts.append(f"Purpose: {knowledge['purpose']}")
        
        # Frameworks
        if knowledge.get('frameworks'):
            frameworks_str = ', '.join(knowledge['frameworks'])
            summary_parts.append(f"Frameworks: {frameworks_str}")
        
        # Structure info
        structure = knowledge.get('structure', {})
        if structure.get('total_files'):
            summary_parts.append(f"Files: {structure['total_files']}")
        
        # Entry points
        if knowledge.get('entry_points'):
            entry_points = ', '.join(knowledge['entry_points'])
            summary_parts.append(f"Entry points: {entry_points}")
        
        if not summary_parts:
            return "Project structure detected but analysis is incomplete."
        
        return ' | '.join(summary_parts)
    
    def get_context_for_ai(self) -> str:
        """Get project context formatted for AI prompts"""
        knowledge = self.project_knowledge
        
        context_parts = [
            "PROJECT CONTEXT:",
            f"- Name: {self.project_path.name}",
        ]
        
        if knowledge.get('primary_language'):
            context_parts.append(f"- Primary Language: {knowledge['primary_language'].title()}")
        
        if knowledge.get('project_type'):
            context_parts.append(f"- Type: {knowledge['project_type'].replace('_', ' ').title()}")
        
        if knowledge.get('purpose'):
            context_parts.append(f"- Purpose: {knowledge['purpose']}")
        
        if knowledge.get('frameworks'):
            frameworks = ', '.join(knowledge['frameworks'])
            context_parts.append(f"- Frameworks: {frameworks}")
        
        if knowledge.get('entry_points'):
            entry_points = ', '.join(knowledge['entry_points'])
            context_parts.append(f"- Entry Points: {entry_points}")
        
        # Add recent interactions for context
        if self.interactions:
            recent_interactions = self.interactions[-3:]  # Last 3 interactions
            context_parts.append("- Recent Work:")
            for interaction in recent_interactions:
                request_summary = interaction['user_request'][:100] + '...' if len(interaction['user_request']) > 100 else interaction['user_request']
                context_parts.append(f"  * {request_summary}")
        
        return '\n'.join(context_parts)