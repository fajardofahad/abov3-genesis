"""
ABOV3 Genesis - Enhanced Assistant with Claude Coder Capabilities
Integrates smart debugging and file management APIs for Claude-level intelligence
"""

import asyncio
import os
import sys
import re
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import existing components
from .assistant_v2 import EnhancedAssistant
from .claude_debugger import ClaudeDebugger, debug_code, handle_error
from .file_management import FileManagementAPI


class ClaudeEnhancedAssistant(EnhancedAssistant):
    """
    ABOV3 Genesis Assistant enhanced with Claude Coder capabilities
    Includes smart debugging, file management, and proactive error prevention
    """
    
    def __init__(self, agent=None, project_context: Dict[str, Any] = None, genesis_engine=None):
        super().__init__(agent, project_context, genesis_engine)
        
        # Initialize Claude Coder components
        self.debugger = ClaudeDebugger()
        self.file_manager = FileManagementAPI(
            project_context.get('project_path') if project_context else None
        )
        
        # Enable Claude-style features
        self.proactive_debugging = True
        self.auto_fix_errors = True
        self.real_time_analysis = True
        self.context_aware_suggestions = True
        
        # Code analysis cache
        self.analysis_cache = {}
        
        # Error prevention system
        self.prevented_errors = []
        
        # File operation hooks
        self.file_hooks = {
            'before_create': [],
            'after_create': [],
            'before_modify': [],
            'after_modify': [],
            'before_delete': [],
            'after_delete': []
        }
        
        # Install global error handler
        self._install_error_handler()
        
        logger.info("Claude Enhanced Assistant initialized with smart debugging and file management")
    
    def _install_error_handler(self):
        """Install global error handler for proactive debugging"""
        def exception_handler(exc_type, exc_value, exc_traceback):
            if self.proactive_debugging:
                asyncio.create_task(self._handle_global_error(exc_value, exc_traceback))
            else:
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = exception_handler
    
    async def _handle_global_error(self, error: Exception, tb):
        """Handle global errors with intelligent debugging"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'traceback': traceback.format_tb(tb)
        }
        
        debug_info = await self.debugger.debug_with_context(error, context)
        
        if self.auto_fix_errors and debug_info.get('suggested_fixes'):
            logger.info(f"Attempting auto-fix for {type(error).__name__}")
            for fix in debug_info['suggested_fixes']:
                await self._apply_fix(fix)
    
    async def process(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """
        Enhanced request processing with Claude Coder capabilities
        """
        try:
            # Perform real-time analysis on user input
            if self.real_time_analysis:
                input_analysis = await self._analyze_user_input(user_input)
                if input_analysis.get('potential_issues'):
                    logger.warning(f"Potential issues detected: {input_analysis['potential_issues']}")
            
            # Check if this is a file operation request
            file_op = self._detect_file_operation(user_input)
            if file_op:
                return await self._handle_file_operation(file_op, user_input)
            
            # Check if this is a debugging request
            if self._is_debug_request(user_input):
                return await self._handle_debug_request(user_input)
            
            # Process with base assistant
            response = await super().process(user_input, context)
            
            # Post-process response with debugging
            if self.proactive_debugging:
                response = await self._enhance_response_with_debugging(response, user_input)
            
            return response
            
        except Exception as e:
            # Handle errors with intelligent debugging
            debug_info = await handle_error(e, {'user_input': user_input, 'context': context})
            
            if self.auto_fix_errors and debug_info.get('suggested_fixes'):
                # Try to apply fixes and retry
                for fix in debug_info['suggested_fixes']:
                    if await self._apply_fix(fix):
                        try:
                            return await super().process(user_input, context)
                        except:
                            pass  # Fix didn't work, continue to error response
            
            # Return detailed error information
            return self._format_debug_response(debug_info)
    
    async def _analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for potential issues"""
        analysis = {
            'intent': 'unknown',
            'potential_issues': [],
            'suggestions': []
        }
        
        # Detect code snippets
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', user_input, re.DOTALL)
        for code in code_blocks:
            code_analysis = await debug_code(code)
            if code_analysis['errors']:
                analysis['potential_issues'].extend(code_analysis['errors'])
            if code_analysis['suggestions']:
                analysis['suggestions'].extend(code_analysis['suggestions'])
        
        return analysis
    
    def _detect_file_operation(self, user_input: str) -> Optional[Dict]:
        """Detect file operation requests"""
        operations = {
            r'create\s+(?:a\s+)?file\s+(?:named\s+)?["\']?([^"\']+)["\']?': 'create',
            r'modify\s+(?:the\s+)?file\s+["\']?([^"\']+)["\']?': 'modify',
            r'delete\s+(?:the\s+)?file\s+["\']?([^"\']+)["\']?': 'delete',
            r'rename\s+["\']?([^"\']+)["\']?\s+to\s+["\']?([^"\']+)["\']?': 'rename',
            r'explain\s+(?:the\s+)?file\s+["\']?([^"\']+)["\']?': 'explain',
            r'search\s+for\s+["\']?([^"\']+)["\']?': 'search',
            r'backup\s+(?:the\s+)?file\s+["\']?([^"\']+)["\']?': 'backup',
            r'restore\s+(?:the\s+)?file\s+["\']?([^"\']+)["\']?': 'restore'
        }
        
        for pattern, operation in operations.items():
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return {
                    'operation': operation,
                    'params': match.groups()
                }
        
        return None
    
    async def _handle_file_operation(self, file_op: Dict, user_input: str) -> str:
        """Handle file operations with Claude Coder style"""
        operation = file_op['operation']
        params = file_op['params']
        
        try:
            if operation == 'create':
                # Extract content if provided
                content = ""
                content_match = re.search(r'with\s+content:\s*```\w*\n(.*?)\n```', user_input, re.DOTALL)
                if content_match:
                    content = content_match.group(1)
                
                # Analyze content before creation
                if content and self.proactive_debugging:
                    analysis = await debug_code(content, params[0])
                    if analysis['errors']:
                        return self._format_analysis_response(analysis, "Found issues in the code")
                
                result = await self.file_manager.create_file(params[0], content)
                
            elif operation == 'modify':
                # Extract changes
                changes = self._extract_changes(user_input)
                
                # Validate changes before applying
                if self.proactive_debugging:
                    current = await self.file_manager.read_file(params[0])
                    if current['success']:
                        analysis = await debug_code(changes if isinstance(changes, str) else current['content'])
                        if analysis['errors']:
                            return self._format_analysis_response(analysis, "Found issues in the changes")
                
                result = await self.file_manager.modify_file(params[0], changes)
                
            elif operation == 'explain':
                result = await self.file_manager.explain_file(params[0])
                
            elif operation == 'delete':
                result = await self.file_manager.delete_file(params[0])
                
            elif operation == 'rename':
                result = await self.file_manager.rename_file(params[0], params[1])
                
            elif operation == 'search':
                content_search = 'content' in user_input.lower()
                result = await self.file_manager.search_files(params[0], content_search)
                
            elif operation == 'backup':
                result = await self.file_manager.backup_file(params[0])
                
            elif operation == 'restore':
                version = None
                version_match = re.search(r'version\s+(\d+)', user_input)
                if version_match:
                    version = int(version_match.group(1))
                result = await self.file_manager.restore_file(params[0], version)
            
            else:
                return f"Unknown file operation: {operation}"
            
            return self._format_file_operation_response(operation, result)
            
        except Exception as e:
            debug_info = await handle_error(e, {'operation': operation, 'params': params})
            return self._format_debug_response(debug_info)
    
    def _is_debug_request(self, user_input: str) -> bool:
        """Check if user is requesting debugging"""
        debug_keywords = [
            'debug', 'fix', 'error', 'issue', 'problem', 'bug',
            'trace', 'analyze', 'diagnose', 'troubleshoot'
        ]
        return any(keyword in user_input.lower() for keyword in debug_keywords)
    
    async def _handle_debug_request(self, user_input: str) -> str:
        """Handle debugging requests"""
        # Extract code or error from input
        code_match = re.search(r'```\w*\n(.*?)\n```', user_input, re.DOTALL)
        error_match = re.search(r'error:?\s*(.*?)(?:\n|$)', user_input, re.IGNORECASE)
        
        if code_match:
            code = code_match.group(1)
            analysis = await debug_code(code)
            return self._format_analysis_response(analysis, "Code Analysis Results")
        
        elif error_match:
            error_text = error_match.group(1)
            # Create a synthetic exception for analysis
            try:
                raise Exception(error_text)
            except Exception as e:
                debug_info = await handle_error(e, {'user_input': user_input})
                return self._format_debug_response(debug_info)
        
        else:
            return "Please provide code or error details for debugging."
    
    async def _enhance_response_with_debugging(self, response: str, user_input: str) -> str:
        """Enhance response with proactive debugging insights"""
        # Check for code in response
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response, re.DOTALL)
        
        if code_blocks:
            enhanced_parts = []
            last_end = 0
            
            for match in re.finditer(r'```(\w+)?\n(.*?)\n```', response, re.DOTALL):
                # Add text before code block
                enhanced_parts.append(response[last_end:match.start()])
                
                lang = match.group(1) or 'python'
                code = match.group(2)
                
                # Analyze code
                analysis = await debug_code(code)
                
                # Add code block
                enhanced_parts.append(f"```{lang}\n{code}\n```")
                
                # Add analysis if issues found
                if analysis['errors'] or analysis['warnings']:
                    enhanced_parts.append("\n\n**Code Analysis:**\n")
                    
                    if analysis['errors']:
                        enhanced_parts.append("⚠️ **Potential Issues:**\n")
                        for error in analysis['errors']:
                            enhanced_parts.append(f"- {error['message']}")
                            if error.get('auto_fix'):
                                enhanced_parts.append(f"\n  Fix: {error['auto_fix']['description']}")
                        enhanced_parts.append("\n")
                    
                    if analysis['warnings']:
                        enhanced_parts.append("💡 **Suggestions:**\n")
                        for warning in analysis['warnings']:
                            enhanced_parts.append(f"- {warning['message']}\n")
                
                last_end = match.end()
            
            # Add remaining text
            enhanced_parts.append(response[last_end:])
            
            return ''.join(enhanced_parts)
        
        return response
    
    async def _apply_fix(self, fix: Dict) -> bool:
        """Apply an auto-fix suggestion"""
        try:
            if fix.get('type') == 'add_definition':
                # Add variable definition
                logger.info(f"Applying fix: {fix['description']}")
                return True
            elif fix.get('type') == 'install_module':
                # Install missing module
                os.system(fix['command'])
                return True
            elif fix.get('type') == 'add_import':
                # Add import statement
                logger.info(f"Applying fix: {fix['description']}")
                return True
            else:
                return False
        except:
            return False
    
    def _extract_changes(self, user_input: str) -> Union[str, Dict, List]:
        """Extract file changes from user input"""
        # Check for direct content replacement
        content_match = re.search(r'```\w*\n(.*?)\n```', user_input, re.DOTALL)
        if content_match:
            return content_match.group(1)
        
        # Check for line-based changes
        changes = []
        for match in re.finditer(r'line\s+(\d+):\s*(.*?)(?:\n|$)', user_input):
            changes.append({
                'action': 'replace',
                'line': int(match.group(1)) - 1,
                'text': match.group(2)
            })
        
        if changes:
            return changes
        
        # Default to full content
        return user_input
    
    def _format_file_operation_response(self, operation: str, result: Dict) -> str:
        """Format file operation response"""
        if result['success']:
            if operation == 'create':
                return f"✅ File created successfully: `{result['path']}`\n- Size: {result['size']} bytes\n- Type: {result['type']}"
            elif operation == 'modify':
                return f"✅ File modified successfully: `{result['path']}`\n- Backup: `{result['backup']}`\n- Lines changed: {result['lines_changed']}"
            elif operation == 'explain':
                return f"📄 **File Analysis: `{result['path']}`**\n\n**Purpose:** {result['purpose']}\n**Type:** {result['type']}\n**Summary:** {result['summary']}\n\n**Structure:**\n{json.dumps(result['structure'], indent=2)}"
            elif operation == 'delete':
                return f"✅ File deleted: `{result['path']}`\n- Backup: `{result['backup']}`"
            elif operation == 'rename':
                response = f"✅ File renamed:\n- From: `{result['old_path']}`\n- To: `{result['new_path']}`"
                if result['imports_updated']:
                    response += f"\n- Updated imports in {len(result['imports_updated'])} files"
                return response
            elif operation == 'search':
                if result['count'] == 0:
                    return f"No files found matching pattern: `{result['pattern']}`"
                else:
                    response = f"Found {result['count']} files matching `{result['pattern']}`:\n"
                    for item in result['results'][:10]:  # Show first 10
                        if result['content_search']:
                            response += f"\n**{item['path']}**\n"
                            for match in item['matches'][:3]:  # Show first 3 matches
                                response += f"  Line {match['line']}: {match['content']}\n"
                        else:
                            response += f"- {item['name']} ({item['size']} bytes)\n"
                    return response
            elif operation == 'backup':
                return f"✅ Backup created:\n- File: `{result['original']}`\n- Backup: `{result['backup']}`"
            elif operation == 'restore':
                return f"✅ File restored:\n- Path: `{result['path']}`\n- From: `{result['restored_from']}`"
        else:
            return f"❌ {operation.capitalize()} failed: {result['error']}\n{result.get('suggestion', '')}"
    
    def _format_analysis_response(self, analysis: Dict, title: str) -> str:
        """Format code analysis response"""
        response = f"## {title}\n\n"
        
        if analysis['errors']:
            response += "### ❌ Errors Found:\n"
            for error in analysis['errors']:
                response += f"- **Line {error.get('line', '?')}:** {error['message']}\n"
                if error.get('auto_fix'):
                    response += f"  - **Fix:** {error['auto_fix']['description']}\n"
                    if error['auto_fix'].get('code'):
                        response += f"    ```\n    {error['auto_fix']['code']}\n    ```\n"
        
        if analysis['warnings']:
            response += "\n### ⚠️ Warnings:\n"
            for warning in analysis['warnings']:
                response += f"- {warning['message']}\n"
                if warning.get('suggestion'):
                    response += f"  - **Suggestion:** {warning['suggestion']}\n"
        
        if analysis['suggestions']:
            response += "\n### 💡 Suggestions:\n"
            for suggestion in analysis['suggestions']:
                response += f"- {suggestion.get('message', suggestion)}\n"
        
        if analysis.get('quality_score'):
            response += f"\n### 📊 Code Quality Score: {analysis['quality_score']}/100\n"
        
        if analysis.get('complexity_analysis'):
            complexity = analysis['complexity_analysis']
            response += f"\n### 📈 Complexity Metrics:\n"
            response += f"- Cyclomatic Complexity: {complexity.get('cyclomatic_complexity', 'N/A')}\n"
            response += f"- Nesting Depth: {complexity.get('nesting_depth', 'N/A')}\n"
        
        if analysis.get('security_issues'):
            response += "\n### 🔒 Security Issues:\n"
            for issue in analysis['security_issues']:
                response += f"- **{issue['severity'].upper()}:** {issue['message']}\n"
                if issue.get('recommendation'):
                    response += f"  - **Recommendation:** {issue['recommendation']}\n"
        
        return response
    
    def _format_debug_response(self, debug_info: Dict) -> str:
        """Format debug information response"""
        response = f"## 🐛 Debug Analysis\n\n"
        response += f"**Error Type:** `{debug_info['error_type']}`\n"
        response += f"**Message:** {debug_info['error_message']}\n\n"
        
        if debug_info.get('root_cause'):
            cause = debug_info['root_cause']
            response += f"### Root Cause:\n"
            response += f"- **File:** `{cause['file']}`\n"
            response += f"- **Line:** {cause['line']}\n"
            response += f"- **Function:** `{cause['function']}`\n"
            response += f"- **Code:** `{cause['code']}`\n\n"
        
        if debug_info.get('suggested_fixes'):
            response += "### 🔧 Suggested Fixes:\n"
            for fix in debug_info['suggested_fixes']:
                response += f"- **{fix['description']}**\n"
                if fix.get('fix'):
                    if fix['fix'].get('code'):
                        response += f"  ```\n  {fix['fix']['code']}\n  ```\n"
                    if fix['fix'].get('alternative'):
                        response += f"  Or: `{fix['fix']['alternative']}`\n"
        
        if debug_info.get('prevention_tips'):
            response += "\n### 💡 Prevention Tips:\n"
            for tip in debug_info['prevention_tips']:
                response += f"- {tip}\n"
        
        if debug_info.get('stack_analysis'):
            response += "\n### 📚 Call Stack:\n"
            for frame in debug_info['stack_analysis']['call_sequence'][:5]:  # Show top 5
                response += f"- `{frame['function']}` at {frame['file']}:{frame['line']}\n"
        
        return response
    
    # File Management API Methods (Claude Coder style)
    
    async def create_file(self, path: str, content: str = "") -> Dict[str, Any]:
        """Create a new file with intelligent validation"""
        # Run before hooks
        for hook in self.file_hooks['before_create']:
            await hook(path, content)
        
        # Analyze content before creation
        if content and self.proactive_debugging:
            analysis = await debug_code(content, path)
            if analysis['errors'] and not self.auto_fix_errors:
                return {
                    'success': False,
                    'error': 'Code contains errors',
                    'analysis': analysis
                }
            elif analysis['errors'] and self.auto_fix_errors:
                # Apply auto-fixes
                for error in analysis['errors']:
                    if error.get('auto_fix'):
                        content = self._apply_content_fix(content, error['auto_fix'])
        
        result = await self.file_manager.create_file(path, content)
        
        # Run after hooks
        for hook in self.file_hooks['after_create']:
            await hook(path, result)
        
        return result
    
    async def modify_file(self, path: str, changes: Union[str, Dict, List]) -> Dict[str, Any]:
        """Modify a file with intelligent validation"""
        # Run before hooks
        for hook in self.file_hooks['before_modify']:
            await hook(path, changes)
        
        result = await self.file_manager.modify_file(path, changes)
        
        # Run after hooks
        for hook in self.file_hooks['after_modify']:
            await hook(path, result)
        
        return result
    
    async def explain_file(self, path: str) -> Dict[str, Any]:
        """Get AI-powered explanation of a file"""
        return await self.file_manager.explain_file(path)
    
    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file with metadata"""
        return await self.file_manager.read_file(path)
    
    async def delete_file(self, path: str) -> Dict[str, Any]:
        """Delete a file with backup"""
        # Run before hooks
        for hook in self.file_hooks['before_delete']:
            await hook(path)
        
        result = await self.file_manager.delete_file(path)
        
        # Run after hooks
        for hook in self.file_hooks['after_delete']:
            await hook(path, result)
        
        return result
    
    async def rename_file(self, old_path: str, new_path: str) -> Dict[str, Any]:
        """Rename a file with import updates"""
        return await self.file_manager.rename_file(old_path, new_path)
    
    async def search_files(self, pattern: str, content_search: bool = False) -> Dict[str, Any]:
        """Search for files by pattern or content"""
        return await self.file_manager.search_files(pattern, content_search)
    
    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get comprehensive file information"""
        return await self.file_manager.get_file_info(path)
    
    async def backup_file(self, path: str) -> Dict[str, Any]:
        """Create a backup of a file"""
        return await self.file_manager.backup_file(path)
    
    async def restore_file(self, path: str, version: Optional[Union[int, str]] = None) -> Dict[str, Any]:
        """Restore a file from backup"""
        return await self.file_manager.restore_file(path, version)
    
    def _apply_content_fix(self, content: str, fix: Dict) -> str:
        """Apply a fix to content"""
        if fix.get('type') == 'add_definition':
            # Add at the beginning
            return fix['code'] + '\n' + content
        elif fix.get('type') == 'add_import':
            # Add import at the top
            lines = content.splitlines()
            import_line = fix['code']
            
            # Find where to insert import
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    insert_pos = i + 1
                elif line.strip() and not line.startswith('#'):
                    break
            
            lines.insert(insert_pos, import_line)
            return '\n'.join(lines)
        
        return content
    
    def add_file_hook(self, event: str, hook: callable):
        """Add a hook for file operations"""
        if event in self.file_hooks:
            self.file_hooks[event].append(hook)
    
    def enable_feature(self, feature: str):
        """Enable a Claude Coder feature"""
        features = {
            'proactive_debugging': 'proactive_debugging',
            'auto_fix': 'auto_fix_errors',
            'real_time': 'real_time_analysis',
            'context_aware': 'context_aware_suggestions'
        }
        
        if feature in features:
            setattr(self, features[feature], True)
            logger.info(f"Enabled feature: {feature}")
    
    def disable_feature(self, feature: str):
        """Disable a Claude Coder feature"""
        features = {
            'proactive_debugging': 'proactive_debugging',
            'auto_fix': 'auto_fix_errors',
            'real_time': 'real_time_analysis',
            'context_aware': 'context_aware_suggestions'
        }
        
        if feature in features:
            setattr(self, features[feature], False)
            logger.info(f"Disabled feature: {feature}")


# Global enhanced assistant instance
_claude_assistant = None

def get_claude_assistant(project_context: Optional[Dict] = None) -> ClaudeEnhancedAssistant:
    """Get or create global Claude-enhanced assistant instance"""
    global _claude_assistant
    if _claude_assistant is None:
        _claude_assistant = ClaudeEnhancedAssistant(project_context=project_context)
    return _claude_assistant