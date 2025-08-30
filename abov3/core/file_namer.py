"""
ABOV3 Genesis - AI-Powered File Naming System
Generates professional, best-practice file and directory names using AI
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import asyncio

class AIFileNamer:
    """
    AI-powered file naming system that generates professional,
    best-practice compliant file and directory structures
    """
    
    def __init__(self, ollama_client=None):
        self.ollama_client = ollama_client
        
        # Cache for naming conventions to avoid repeated AI calls
        self.naming_cache = {}
        
        # Default naming conventions as fallback
        self.default_conventions = {
            'html': {
                'landing_page': 'index.html',
                'about': 'about.html',
                'contact': 'contact.html',
                'portfolio': 'portfolio.html',
                'blog': 'blog.html',
                'services': 'services.html'
            },
            'css': {
                'main': 'styles.css',
                'components': 'components.css',
                'layout': 'layout.css',
                'theme': 'theme.css',
                'utilities': 'utilities.css'
            },
            'javascript': {
                'main': 'main.js',
                'app': 'app.js',
                'utils': 'utils.js',
                'config': 'config.js',
                'api': 'api.js',
                'components': 'components.js'
            },
            'python': {
                'main': 'main.py',
                'app': 'app.py',
                'utils': 'utils.py',
                'config': 'config.py',
                'models': 'models.py',
                'views': 'views.py',
                'tests': 'test_main.py'
            },
            'directories': {
                'styles': 'css',
                'scripts': 'js',
                'images': 'assets/images',
                'fonts': 'assets/fonts',
                'components': 'components',
                'pages': 'pages',
                'source': 'src',
                'public': 'public',
                'static': 'static',
                'templates': 'templates'
            }
        }
    
    async def get_file_names(
        self, 
        code_blocks: List[Dict[str, str]], 
        user_input: str,
        project_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get AI-recommended file names and directory structure for code blocks
        
        Args:
            code_blocks: List of code blocks with language and content
            user_input: Original user request
            project_type: Optional project type (website, api, app, etc.)
            
        Returns:
            List of dictionaries with recommended file paths and metadata
        """
        if not self.ollama_client:
            return self._get_default_names(code_blocks, user_input)
        
        # Check cache first
        cache_key = f"{user_input[:50]}_{len(code_blocks)}"
        if cache_key in self.naming_cache:
            return self.naming_cache[cache_key]
        
        try:
            # Prepare prompt for AI
            prompt = self._build_naming_prompt(code_blocks, user_input, project_type)
            
            # Get AI recommendation
            response = await self._get_ai_naming_suggestion(prompt)
            
            # Parse AI response
            file_structure = self._parse_ai_response(response, code_blocks)
            
            # Cache the result
            self.naming_cache[cache_key] = file_structure
            
            return file_structure
            
        except Exception as e:
            print(f"[DEBUG] Error getting AI file names: {e}")
            return self._get_default_names(code_blocks, user_input)
    
    def _build_naming_prompt(
        self, 
        code_blocks: List[Dict[str, str]], 
        user_input: str,
        project_type: Optional[str]
    ) -> str:
        """Build prompt for AI to suggest file names"""
        
        # Analyze code blocks
        languages = [block.get('language', 'unknown') for block in code_blocks]
        
        # Extract key features from code
        features = []
        for block in code_blocks[:3]:  # Analyze first 3 blocks
            code_snippet = block['code'][:500]  # First 500 chars
            if 'class' in code_snippet or 'function' in code_snippet:
                features.append('contains functions/classes')
            if '<html' in code_snippet.lower():
                features.append('HTML page')
            if 'import' in code_snippet or 'require' in code_snippet:
                features.append('has dependencies')
        
        prompt = f"""You are a senior developer expert in file naming conventions and project structure.

User Request: "{user_input}"
Project Type: {project_type or 'general web project'}
Number of code blocks: {len(code_blocks)}
Languages detected: {', '.join(set(languages))}
Code features: {', '.join(features) if features else 'standard code'}

Based on this information, suggest professional file names and directory structure following these best practices:
1. Use lowercase with hyphens for file names (kebab-case)
2. Use meaningful, descriptive names
3. Follow language-specific conventions
4. Organize files into appropriate directories
5. Consider scalability and maintainability

Respond with a JSON structure like this:
{{
    "files": [
        {{"index": 0, "path": "index.html", "description": "Main landing page"}},
        {{"index": 1, "path": "css/styles.css", "description": "Main stylesheet"}},
        {{"index": 2, "path": "js/main.js", "description": "Main JavaScript file"}}
    ],
    "structure_explanation": "Brief explanation of the naming choices"
}}

Provide ONLY the JSON response, no additional text."""
        
        return prompt
    
    async def _get_ai_naming_suggestion(self, prompt: str) -> str:
        """Get naming suggestion from AI"""
        try:
            if not await self.ollama_client.is_available():
                return "{}"
            
            # Use a code-optimized model if available
            models = await self.ollama_client.list_models()
            model_names = [m.get('name', '') for m in models]
            
            # Prefer code-specific models
            preferred_models = ['deepseek-coder', 'codellama', 'codegemma', 'llama3']
            model = 'llama3'  # default
            
            for pref in preferred_models:
                if any(pref in name for name in model_names):
                    model = next(name for name in model_names if pref in name)
                    break
            
            # Get response
            messages = [
                {"role": "system", "content": "You are an expert in software development best practices and file naming conventions. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            response_text = ""
            async for chunk in self.ollama_client.chat(model, messages, stream=False):
                if "message" in chunk:
                    response_text = chunk["message"].get("content", "")
                    break
            
            return response_text
            
        except Exception as e:
            print(f"[DEBUG] Error getting AI naming suggestion: {e}")
            return "{}"
    
    def _parse_ai_response(self, response: str, code_blocks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Parse AI response and create file structure"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return self._get_default_names(code_blocks, "")
            
            data = json.loads(json_match.group())
            
            if 'files' not in data or not isinstance(data['files'], list):
                return self._get_default_names(code_blocks, "")
            
            # Build file structure
            file_structure = []
            for item in data['files']:
                index = item.get('index', 0)
                if index < len(code_blocks):
                    file_structure.append({
                        'path': item.get('path', f'file_{index}.txt'),
                        'code': code_blocks[index]['code'],
                        'language': code_blocks[index].get('language', 'text'),
                        'description': item.get('description', 'Generated file')
                    })
            
            # Add any remaining code blocks with default names
            for i in range(len(file_structure), len(code_blocks)):
                language = code_blocks[i].get('language', 'text')
                extension = self._get_extension(language)
                file_structure.append({
                    'path': f'additional_{i}{extension}',
                    'code': code_blocks[i]['code'],
                    'language': language,
                    'description': 'Additional generated file'
                })
            
            return file_structure
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[DEBUG] Error parsing AI response: {e}")
            return self._get_default_names(code_blocks, "")
    
    def _get_default_names(self, code_blocks: List[Dict[str, str]], user_input: str) -> List[Dict[str, Any]]:
        """Get default file names when AI is not available"""
        file_structure = []
        user_input_lower = user_input.lower()
        
        for i, block in enumerate(code_blocks):
            language = block.get('language', 'text').lower()
            extension = self._get_extension(language)
            
            # Determine base name from context
            if i == 0:  # First file is usually the main file
                if language == 'html':
                    base_name = 'index'
                elif language in ['python', 'javascript', 'java']:
                    base_name = 'main'
                elif language == 'css':
                    base_name = 'styles'
                else:
                    base_name = 'file'
            else:
                # Subsequent files
                if language == 'css':
                    base_name = f'styles_{i}' if i > 1 else 'styles'
                elif language in ['javascript', 'js']:
                    base_name = f'script_{i}' if i > 1 else 'script'
                else:
                    base_name = f'file_{i}'
            
            # Check for specific patterns in user input
            if 'coffee' in user_input_lower and 'shop' in user_input_lower:
                if language == 'html' and i == 0:
                    base_name = 'index'
                elif language == 'css':
                    base_name = 'coffee-shop-styles'
                elif language in ['javascript', 'js']:
                    base_name = 'coffee-shop'
            elif 'landing' in user_input_lower and 'page' in user_input_lower:
                if language == 'html' and i == 0:
                    base_name = 'index'
                elif language == 'css':
                    base_name = 'landing-page'
            elif 'portfolio' in user_input_lower:
                if language == 'html' and i == 0:
                    base_name = 'portfolio'
                elif language == 'css':
                    base_name = 'portfolio-styles'
            
            # Determine directory structure
            if language == 'css':
                path = f'css/{base_name}{extension}'
            elif language in ['javascript', 'js', 'typescript', 'ts']:
                path = f'js/{base_name}{extension}'
            elif language in ['jpg', 'jpeg', 'png', 'gif', 'svg']:
                path = f'assets/images/{base_name}{extension}'
            else:
                path = f'{base_name}{extension}'
            
            file_structure.append({
                'path': path,
                'code': block['code'],
                'language': language,
                'description': f'Generated {language} file'
            })
        
        return file_structure
    
    def _get_extension(self, language: str) -> str:
        """Get file extension for a language"""
        extensions = {
            'html': '.html', 'htm': '.html',
            'css': '.css', 'scss': '.scss', 'sass': '.sass',
            'javascript': '.js', 'js': '.js', 'jsx': '.jsx',
            'typescript': '.ts', 'ts': '.ts', 'tsx': '.tsx',
            'python': '.py', 'py': '.py',
            'java': '.java', 'kotlin': '.kt', 'kt': '.kt',
            'cpp': '.cpp', 'c++': '.cpp', 'c': '.c',
            'csharp': '.cs', 'cs': '.cs', 'c#': '.cs',
            'php': '.php', 'ruby': '.rb', 'rb': '.rb',
            'go': '.go', 'golang': '.go',
            'rust': '.rs', 'rs': '.rs',
            'swift': '.swift', 'dart': '.dart',
            'sql': '.sql', 'json': '.json', 'xml': '.xml',
            'yaml': '.yaml', 'yml': '.yml',
            'bash': '.sh', 'sh': '.sh', 'shell': '.sh',
            'powershell': '.ps1', 'ps1': '.ps1',
            'markdown': '.md', 'md': '.md',
            'text': '.txt', 'txt': '.txt'
        }
        return extensions.get(language.lower(), '.txt')
    
    def suggest_project_structure(self, project_type: str) -> Dict[str, List[str]]:
        """Suggest a complete project directory structure"""
        structures = {
            'website': {
                'directories': ['css', 'js', 'assets/images', 'assets/fonts'],
                'files': ['index.html', 'css/styles.css', 'js/main.js', 'README.md']
            },
            'webapp': {
                'directories': ['src', 'public', 'src/components', 'src/styles', 'src/utils'],
                'files': ['src/index.js', 'src/App.js', 'public/index.html', 'package.json']
            },
            'api': {
                'directories': ['src', 'src/routes', 'src/models', 'src/controllers', 'tests'],
                'files': ['src/index.js', 'src/config.js', 'package.json', '.env.example']
            },
            'python-app': {
                'directories': ['src', 'tests', 'docs', 'config'],
                'files': ['src/main.py', 'src/__init__.py', 'requirements.txt', 'README.md']
            }
        }
        
        return structures.get(project_type, structures['website'])