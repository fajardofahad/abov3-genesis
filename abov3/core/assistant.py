"""
ABOV3 Genesis - Core Assistant
The main AI assistant that processes user requests and coordinates with other systems
"""

import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import json

from .ollama_client import OllamaClient
from .code_generator import CodeGenerator

class Assistant:
    """
    ABOV3 Genesis Core Assistant
    Processes user requests and coordinates Genesis workflows
    """
    
    def __init__(self, agent=None, project_context: Dict[str, Any] = None, genesis_engine=None):
        self.agent = agent
        self.project_context = project_context or {}
        self.genesis_engine = genesis_engine
        self.ollama_client = OllamaClient()
        
        # Conversation history
        self.conversation_history = []
        
        # Default model if agent doesn't specify one  
        # This will be updated by the main app with the saved model
        self.default_model = "llama3:latest"
        
        # Initialize code generator if we have project context
        self.code_generator = None
        if project_context and 'project_path' in project_context:
            from pathlib import Path
            self.code_generator = CodeGenerator(Path(project_context['project_path']))
    
    async def process(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process user input and return response"""
        try:
            # Update context if provided
            if context:
                self.project_context.update(context)
                
                # Initialize or update code generator if we have project path
                if 'project_path' in context and not self.code_generator:
                    from pathlib import Path
                    self.code_generator = CodeGenerator(Path(context['project_path']))
                    print(f"[DEBUG] Initialized CodeGenerator with path: {context['project_path']}")
                elif 'project_path' in context and self.code_generator:
                    # Update code generator if project path changed
                    from pathlib import Path
                    new_path = Path(context['project_path'])
                    if new_path != self.code_generator.project_path:
                        self.code_generator = CodeGenerator(new_path)
                        print(f"[DEBUG] Updated CodeGenerator with new path: {context['project_path']}")
                else:
                    print(f"[DEBUG] Context: {context}, Code generator exists: {self.code_generator is not None}")
            
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if we need to switch agents automatically
            await self._auto_switch_agent_if_needed(user_input, context)
            
            # Get model and system prompt from agent
            model = self.agent.model if self.agent else self.default_model
            system_prompt = self._build_system_prompt()
            
            # Check if Ollama is available
            if not await self.ollama_client.is_available():
                return self._get_fallback_response(user_input)
            
            # Prepare messages for chat
            messages = self._prepare_messages(user_input, system_prompt)
            
            # Check for different types of requests in priority order
            is_file_operation = self._is_file_operation_request(user_input)
            is_code_request = self._is_code_generation_request(user_input)
            
            print(f"[DEBUG] File operation request: {is_file_operation}")
            print(f"[DEBUG] Code generation request: {is_code_request}")
            
            if is_file_operation:
                response_text = await self._handle_file_operations(user_input)
            elif is_code_request:
                response_text = await self._handle_code_generation(user_input, messages, model)
            else:
                # Get AI response
                response_text = await self._get_ai_response(model, messages)
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant", 
                "content": response_text,
                "timestamp": datetime.now().isoformat(),
                "model": model
            })
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return self._get_error_response(str(e))
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with context"""
        base_prompt = """You are ABOV3 Genesis, an AI coding assistant that transforms ideas into built reality.

Your core mission: Help users go from ideas to working applications through the Genesis workflow:
💡 Idea → 📐 Design → 🔨 Build → 🧪 Test → 🚀 Deploy

PERSONALITY & STYLE:
- Be helpful, enthusiastic, and supportive
- Use clear, practical language
- Focus on actionable solutions
- Encourage users throughout their journey
- Celebrate achievements with appropriate excitement

CAPABILITIES:
- Transform vague ideas into specific implementations
- Generate production-ready code
- Provide architectural guidance
- Debug and optimize applications
- Guide through entire development lifecycle"""
        
        # Add agent-specific context
        if self.agent and hasattr(self.agent, 'system_prompt') and self.agent.system_prompt:
            base_prompt += f"\n\nSPECIALIZATION:\n{self.agent.system_prompt}"
        
        # Add project context
        if self.project_context:
            project_info = self.project_context.get('project', {})
            if project_info:
                context_info = f"""
PROJECT CONTEXT:
- Name: {project_info.get('name', 'Unknown')}
- Language: {project_info.get('language', 'unknown')}
- Framework: {project_info.get('framework', 'unknown')}
- Type: {'Genesis Project' if project_info.get('is_genesis') else 'Regular Project'}
"""
                if project_info.get('is_genesis'):
                    genesis_info = self.project_context.get('genesis', {})
                    if genesis_info.get('idea'):
                        context_info += f"- Original Idea: {genesis_info['idea']}\n"
                    if genesis_info.get('current_phase'):
                        context_info += f"- Current Phase: {genesis_info['current_phase']}\n"
                
                base_prompt += context_info
        
        base_prompt += """
RESPONSE GUIDELINES:
- Be concise but comprehensive
- Provide working code examples when relevant
- Explain your reasoning
- Suggest next steps
- Ask clarifying questions when needed"""
        
        return base_prompt
    
    def _prepare_messages(self, user_input: str, system_prompt: str) -> List[Dict[str, str]]:
        """Prepare messages for the AI model"""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add relevant conversation history (last 10 exchanges)
        recent_history = self.conversation_history[-20:]  # Last 10 user+assistant pairs
        for msg in recent_history:
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user input (if not already in history)
        if not recent_history or recent_history[-1]["content"] != user_input:
            messages.append({"role": "user", "content": user_input})
        
        return messages
    
    async def _get_ai_response(self, model: str, messages: List[Dict[str, str]]) -> str:
        """Get response from AI model"""
        response_parts = []
        
        try:
            async with self.ollama_client:
                # Check if model exists
                if not await self.ollama_client.check_model_exists(model):
                    # Try to use a fallback model
                    available_models = await self.ollama_client.list_models()
                    if available_models:
                        model = available_models[0].get('name', self.default_model)
                    else:
                        return "No AI models available. Please install Ollama and pull a model first."
                
                # Get optimized options based on task type
                task_type = self._detect_task_type(messages[-1]["content"])
                options = self.ollama_client.get_genesis_optimized_options(task_type)
                
                # Stream the response
                async for chunk in self.ollama_client.chat(model, messages, options=options, stream=False):
                    if "error" in chunk:
                        return f"AI Error: {chunk['error']}"
                    
                    if "message" in chunk:
                        response_parts.append(chunk["message"].get("content", ""))
                    elif "response" in chunk:
                        response_parts.append(chunk["response"])
                    
                    if chunk.get("done", False):
                        break
        
        except Exception as e:
            return f"Error communicating with AI: {str(e)}"
        
        return "".join(response_parts).strip()
    
    def _detect_task_type(self, user_input: str) -> str:
        """Detect the type of task from user input"""
        user_input_lower = user_input.lower()
        
        # Code-related keywords
        code_keywords = [
            "code", "function", "class", "implement", "build", "create",
            "write", "generate", "refactor", "debug", "fix", "bug"
        ]
        
        # Creative keywords
        creative_keywords = [
            "idea", "brainstorm", "creative", "design", "concept", "imagine"
        ]
        
        # Analysis keywords
        analysis_keywords = [
            "analyze", "explain", "review", "compare", "evaluate", "assess"
        ]
        
        if any(keyword in user_input_lower for keyword in code_keywords):
            return "code_generation"
        elif any(keyword in user_input_lower for keyword in creative_keywords):
            return "creative_writing"
        elif any(keyword in user_input_lower for keyword in analysis_keywords):
            return "analysis"
        else:
            return "conversation"
    
    def _get_fallback_response(self, user_input: str) -> str:
        """Get a fallback response when AI is not available"""
        return f"""🤖 Ollama AI is not available right now, but I can still help!

Your request: "{user_input}"

To get AI-powered assistance:
1. Make sure Ollama is installed and running
2. Pull a recommended model like: `ollama pull llama3`
3. Start Ollama server: `ollama serve`

For now, here are some general suggestions:
- If you're starting a new project, use `abov3 --new` 
- If you want to see available commands, type `/help`
- If you have a Genesis idea, I can help you structure it manually

Would you like me to help you set up Ollama or provide manual guidance?"""
    
    def _get_error_response(self, error: str) -> str:
        """Get an error response"""
        return f"""❌ Something went wrong while processing your request.

Error details: {error}

Don't worry! Here's what you can try:
1. Check if Ollama is running: `ollama list`
2. Try a simpler request first
3. Restart ABOV3 Genesis if the issue persists
4. Type `/help` for available commands

I'm still here to help with manual guidance and project management!"""
    
    async def process_genesis_command(self, command: str) -> str:
        """Process Genesis-specific commands"""
        if not self.genesis_engine:
            return "Genesis engine not available for this project."
        
        command_lower = command.lower().strip()
        
        if command_lower in ["build my idea", "start genesis", "transform my idea"]:
            return await self._start_genesis_workflow()
        elif command_lower in ["continue genesis", "next phase"]:
            return await self._continue_genesis()
        elif command_lower in ["genesis status", "show progress"]:
            return await self._show_genesis_status()
        else:
            return await self.process(command)
    
    async def _start_genesis_workflow(self) -> str:
        """Start the Genesis workflow"""
        try:
            current_phase = await self.genesis_engine.get_current_phase()
            
            if current_phase == "complete":
                return "🎉 Your Genesis is already complete! Your idea has been transformed into reality."
            
            # Process the current phase
            result = await self.genesis_engine.process_phase(current_phase)
            
            if "error" in result:
                return f"❌ Genesis error: {result['error']}"
            
            phase_icon = self._get_phase_icon(current_phase)
            next_phase = result.get("next_phase")
            
            response = f"{phase_icon} **{current_phase.capitalize()} Phase Complete!**\n\n"
            
            if next_phase:
                next_icon = self._get_phase_icon(next_phase)
                response += f"Ready for {next_icon} **{next_phase.capitalize()} Phase**\n"
                response += f"Type 'continue genesis' to proceed!"
            else:
                response += "🎉 **Genesis Complete!** Your idea is now reality!"
            
            return response
            
        except Exception as e:
            return f"Error in Genesis workflow: {str(e)}"
    
    async def _continue_genesis(self) -> str:
        """Continue the Genesis workflow to the next phase"""
        return await self._start_genesis_workflow()  # Same logic for now
    
    async def _show_genesis_status(self) -> str:
        """Show Genesis status and progress"""
        try:
            stats = await self.genesis_engine.get_genesis_stats()
            
            progress = stats.get('progress_percentage', 0)
            current_phase = stats.get('current_phase', 'unknown')
            idea = stats.get('idea', 'No idea recorded')
            
            status_response = f"""📊 **Genesis Status Report**

💡 **Original Idea**: {idea}

📍 **Current Phase**: {self._get_phase_icon(current_phase)} {current_phase.capitalize()}

📈 **Progress**: {progress:.0f}% Complete

🎯 **Phases**:"""
            
            phases = stats.get('phases', {})
            for phase_name, phase_data in phases.items():
                icon = self._get_phase_icon(phase_name)
                status = phase_data.get('status', 'pending')
                status_icon = self._get_status_icon(status)
                status_response += f"\n  {icon} {phase_name.capitalize()}: {status_icon}"
            
            if stats.get('is_genesis_complete'):
                status_response += "\n\n🎉 **Genesis Complete!** From idea to reality - mission accomplished!"
            else:
                status_response += f"\n\nType 'continue genesis' to advance to the next phase!"
            
            return status_response
            
        except Exception as e:
            return f"Error getting Genesis status: {str(e)}"
    
    def _get_phase_icon(self, phase: str) -> str:
        """Get icon for a Genesis phase"""
        icons = {
            'idea': '💡',
            'design': '📐',
            'build': '🔨',
            'test': '🧪',
            'deploy': '🚀',
            'complete': '✅'
        }
        return icons.get(phase.lower(), '📁')
    
    def _get_status_icon(self, status: str) -> str:
        """Get icon for a status"""
        icons = {
            'complete': '✅',
            'in_progress': '⏳',
            'pending': '⏸️',
            'failed': '❌'
        }
        return icons.get(status.lower(), '⏸️')
    
    def add_context(self, key: str, value: Any):
        """Add context information"""
        if 'dynamic_context' not in self.project_context:
            self.project_context['dynamic_context'] = {}
        self.project_context['dynamic_context'][key] = value
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
    
    def set_agent(self, agent):
        """Set the current agent"""
        self.agent = agent
    
    def set_project_context(self, context: Dict[str, Any]):
        """Set project context"""
        self.project_context = context
    
    async def suggest_next_steps(self) -> List[str]:
        """Suggest next steps based on current context"""
        suggestions = []
        
        # Genesis-specific suggestions
        if self.genesis_engine:
            try:
                current_phase = await self.genesis_engine.get_current_phase()
                if current_phase != "complete":
                    suggestions.append(f"Continue Genesis workflow: {self._get_phase_icon(current_phase)} {current_phase}")
            except:
                pass
        
        # Project-specific suggestions
        project_info = self.project_context.get('project', {})
        if project_info:
            language = project_info.get('language', 'unknown')
            framework = project_info.get('framework', 'unknown')
            
            if language != 'unknown':
                suggestions.append(f"Generate {language} code for your project")
            
            if framework != 'unknown':
                suggestions.append(f"Create {framework} components or modules")
        
        # General suggestions
        suggestions.extend([
            "Ask me to explain any concept or code",
            "Request help with debugging or optimization",
            "Get suggestions for project architecture",
            "Generate tests for your code"
        ])
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _is_code_generation_request(self, user_input: str) -> bool:
        """Check if user input is requesting code generation"""
        user_input_lower = user_input.lower()
        
        # Direct code generation keywords
        code_generation_keywords = [
            'write', 'create', 'generate', 'build', 'make',
            'code', 'function', 'class', 'file', 'script',
            'implement', 'develop', 'program'
        ]
        
        # Programming language names
        language_keywords = [
            'python', 'javascript', 'java', 'c++', 'cpp', 'c#', 'csharp', 'rust', 'go', 'golang',
            'html', 'css', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'typescript', 'dart',
            'shell', 'bash', 'powershell', 'sql', 'r', 'julia', 'lua', 'perl', 'haskell',
            'elixir', 'erlang', 'clojure', 'assembly', 'solidity', 'verilog', 'vhdl'
        ]
        
        # File-related keywords
        file_keywords = [
            'save to', 'write to', 'create file', 'generate file',
            'put in', 'add to project', 'create in'
        ]
        
        # Check for combinations
        has_code_keyword = any(keyword in user_input_lower for keyword in code_generation_keywords)
        has_file_keyword = any(keyword in user_input_lower for keyword in file_keywords)
        has_language_keyword = any(lang in user_input_lower for lang in language_keywords)
        
        # Explicit file extensions - comprehensive list for all coding languages
        coding_extensions = [
            # Web Development
            '.html', '.htm', '.css', '.scss', '.sass', '.less', '.js', '.jsx', '.ts', '.tsx', 
            '.vue', '.svelte', '.php', '.asp', '.aspx', '.jsp',
            
            # Python
            '.py', '.pyw', '.pyx', '.pyi', '.ipynb',
            
            # JavaScript/Node.js
            '.mjs', '.cjs', '.json', '.jsonc',
            
            # Java/JVM
            '.java', '.kt', '.scala', '.clj', '.groovy',
            
            # C/C++
            '.c', '.cc', '.cpp', '.cxx', '.c++', '.h', '.hpp', '.hxx', '.h++',
            
            # C#/.NET
            '.cs', '.vb', '.fs', '.fsx',
            
            # Systems Programming
            '.rs', '.go', '.zig', '.d', '.nim',
            
            # Mobile Development
            '.swift', '.m', '.mm', '.dart',
            
            # Scripting
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
            
            # Data/Config
            '.sql', '.yaml', '.yml', '.toml', '.xml', '.ini', '.cfg', '.conf',
            
            # Other Popular Languages
            '.rb', '.pl', '.pm', '.r', '.R', '.jl', '.lua', '.tcl',
            '.hs', '.lhs', '.elm', '.ml', '.mli', '.ocaml',
            '.erl', '.ex', '.exs', '.clj', '.cljs', '.cljc',
            
            # Assembly
            '.asm', '.s', '.S',
            
            # Specialized
            '.sol', '.v', '.sv', '.vhd', '.vhdl', '.tex', '.md', '.rst'
        ]
        
        has_file_extension = any(ext in user_input_lower for ext in coding_extensions)
        
        has_make_phrase = any(phrase in user_input_lower for phrase in ['write a', 'create a', 'generate a', 'make a'])
        
        print(f"[DEBUG] Code detection - has_code_keyword: {has_code_keyword}, has_file_keyword: {has_file_keyword}, has_file_extension: {has_file_extension}, has_make_phrase: {has_make_phrase}, has_language_keyword: {has_language_keyword}")
        
        # Enhanced detection logic - any of these patterns should trigger code generation:
        # 1. Code keyword + file keyword (e.g., "write code to file")
        # 2. Code keyword + language (e.g., "write python code")
        # 3. File extension mentioned (e.g., "create .html file")
        # 4. Make/create phrase (e.g., "make a script")
        # 5. Language keyword alone (e.g., "python hello world")
        return (has_code_keyword and has_file_keyword) or \
               (has_code_keyword and has_language_keyword) or \
               has_file_extension or \
               has_make_phrase or \
               has_language_keyword
    
    async def _handle_code_generation(self, user_input: str, messages: List[Dict[str, str]], model: str) -> str:
        """Handle code generation requests with file creation and modification"""
        if not self.code_generator:
            # Debug information
            project_path = self.project_context.get('project_path', 'Not set')
            return f"❌ Code generation is not available. Debug info - Project path: {project_path}, Code generator: {self.code_generator is not None}"
        
        try:
            # Check if this is a modification request for existing files
            is_modification_request = any(word in user_input.lower() for word in [
                'update', 'modify', 'change', 'edit', 'fix', 'make', 'set', 'color', 'style', 'add'
            ])
            
            if is_modification_request:
                return await self._handle_file_modification(user_input, messages, model)
            else:
                return await self._handle_new_file_creation(user_input, messages, model)
                
        except Exception as e:
            return f"❌ Error during code generation: {str(e)}"
    
    async def _handle_file_modification(self, user_input: str, messages: List[Dict[str, str]], model: str) -> str:
        """Handle modification of existing files like Claude does"""
        import os
        from pathlib import Path
        
        try:
            # Find existing files that match the context
            project_path = Path(self.code_generator.project_path)
            relevant_files = []
            
            # Look for different file types based on context
            context_lower = user_input.lower()
            if any(word in context_lower for word in ['website', 'web', 'html', 'page']):
                # Look for HTML, CSS, JS files
                file_patterns = ['*.html', '*.htm', '*.css', '*.js']
            elif any(word in context_lower for word in ['script', 'python', 'py']):
                file_patterns = ['*.py']
            elif any(word in context_lower for word in ['style', 'css']):
                file_patterns = ['*.css', '*.html', '*.htm']
            else:
                # General search for common web files
                file_patterns = ['*.html', '*.htm', '*.css', '*.js', '*.py']
            
            # Find matching files
            for pattern in file_patterns:
                try:
                    for file_path in project_path.glob(pattern):
                        if file_path.is_file():
                            relevant_files.append(file_path)
                except:
                    pass
            
            if not relevant_files:
                print("[DEBUG] No existing files found for modification, creating new files")
                return await self._handle_new_file_creation(user_input, messages, model)
            
            print(f"[DEBUG] Found {len(relevant_files)} files for potential modification")
            
            # Read existing files and create modification context
            file_contents = {}
            for file_path in relevant_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_contents[file_path.name] = {
                            'path': file_path,
                            'content': content
                        }
                        print(f"[DEBUG] Read existing file: {file_path.name} ({len(content)} chars)")
                except Exception as e:
                    print(f"[DEBUG] Could not read {file_path}: {e}")
            
            if not file_contents:
                return await self._handle_new_file_creation(user_input, messages, model)
            
            # Create enhanced prompt for AI with existing file contents
            modification_prompt = f"""
I need to modify existing files in a project. Here are the current files and their contents:

{chr(10).join([f"**{filename}:**{chr(10)}```{chr(10)}{info['content']}{chr(10)}```{chr(10)}" for filename, info in file_contents.items()])}

User Request: {user_input}

Please provide the complete modified file content(s) with the requested changes. Maintain the existing structure and only change what's necessary for the request. Format your response with clear code blocks showing the full updated file contents.
"""
            
            # Get AI response for modifications
            modification_messages = messages[:-1] + [{"role": "user", "content": modification_prompt}]
            ai_response = await self._get_ai_response(model, modification_messages)
            
            # Extract code blocks from AI response
            code_blocks = self._extract_code_blocks(ai_response)
            
            if not code_blocks:
                return f"❌ AI couldn't generate modifications for the request: {user_input}"
            
            print(f"[DEBUG] AI generated {len(code_blocks)} modified code blocks")
            
            # Apply modifications to files
            modified_files = []
            for i, code_block in enumerate(code_blocks):
                # Try to match code block to appropriate file
                target_file = None
                language = code_block.get('language', '').lower()
                
                # Smart file matching based on language and content
                if language in ['html', 'htm'] or '<html' in code_block['code'].lower():
                    target_file = next((info for name, info in file_contents.items() if name.endswith(('.html', '.htm'))), None)
                elif language == 'css' or ('body' in code_block['code'] and '{' in code_block['code']):
                    target_file = next((info for name, info in file_contents.items() if name.endswith('.css')), None)
                elif language in ['javascript', 'js']:
                    target_file = next((info for name, info in file_contents.items() if name.endswith('.js')), None)
                elif language == 'python':
                    target_file = next((info for name, info in file_contents.items() if name.endswith('.py')), None)
                
                # If no specific match, use the first file
                if not target_file and file_contents:
                    target_file = list(file_contents.values())[0]
                
                if target_file:
                    # Write the modified content back to the file
                    try:
                        with open(target_file['path'], 'w', encoding='utf-8') as f:
                            f.write(code_block['code'])
                        
                        modified_files.append({
                            'path': target_file['path'].name,
                            'full_path': str(target_file['path']),
                            'size': len(code_block['code']),
                            'lines': code_block['code'].count('\n') + 1
                        })
                        print(f"[DEBUG] Successfully modified: {target_file['path'].name}")
                    except Exception as e:
                        print(f"[DEBUG] Failed to modify {target_file['path'].name}: {e}")
            
            # Prepare response
            if modified_files:
                file_summary = "\n\n📝 **Files Modified:**\n"
                for file_info in modified_files:
                    file_summary += f"✅ `{file_info['path']}` ({file_info['lines']} lines, {file_info['size']} bytes)\n"
                
                ai_response += file_summary
                ai_response += "\n🎯 **Your files have been updated with the requested changes!**"
            
            return ai_response
            
        except Exception as e:
            return f"❌ Error during file modification: {str(e)}"
    
    async def _handle_new_file_creation(self, user_input: str, messages: List[Dict[str, str]], model: str) -> str:
        """Handle creation of new files"""
        try:
            # Get AI response for the code
            ai_response = await self._get_ai_response(model, messages)
            
            # Try to extract code blocks and file paths from the response
            code_blocks = self._extract_code_blocks(ai_response)
            file_paths = self._extract_file_paths(user_input)
            
            print(f"[DEBUG] Found {len(code_blocks)} code blocks")
            print(f"[DEBUG] Found {len(file_paths)} file paths")
            
            # Always create files if we have code blocks, even without explicit file paths
            if code_blocks:
                print(f"[DEBUG] Processing {len(code_blocks)} code blocks for file creation")
                created_files = []
                
                for i, code_block in enumerate(code_blocks):
                    # Determine file path
                    if i < len(file_paths):
                        file_path = file_paths[i]
                    else:
                        # Generate file path based on language - comprehensive mapping
                        language = code_block.get('language', 'python')
                        extensions = {
                            # Web Development
                            'html': '.html',
                            'htm': '.html',
                            'css': '.css',
                            'scss': '.scss',
                            'sass': '.sass',
                            'less': '.less',
                            'javascript': '.js',
                            'js': '.js',
                            'jsx': '.jsx',
                            'typescript': '.ts',
                            'ts': '.ts',
                            'tsx': '.tsx',
                            'vue': '.vue',
                            'svelte': '.svelte',
                            'php': '.php',
                            'asp': '.asp',
                            'aspx': '.aspx',
                            'jsp': '.jsp',
                            
                            # Python
                            'python': '.py',
                            'py': '.py',
                            'pyx': '.pyx',
                            'pyi': '.pyi',
                            
                            # Java/JVM Languages
                            'java': '.java',
                            'kotlin': '.kt',
                            'kt': '.kt',
                            'scala': '.scala',
                            'clojure': '.clj',
                            'clj': '.clj',
                            'groovy': '.groovy',
                            
                            # C/C++
                            'c': '.c',
                            'cpp': '.cpp',
                            'c++': '.cpp',
                            'cxx': '.cxx',
                            'cc': '.cc',
                            'h': '.h',
                            'hpp': '.hpp',
                            'hxx': '.hxx',
                            
                            # C#/.NET
                            'csharp': '.cs',
                            'cs': '.cs',
                            'c#': '.cs',
                            'vb': '.vb',
                            'vbnet': '.vb',
                            'fsharp': '.fs',
                            'fs': '.fs',
                            
                            # Systems Programming
                            'rust': '.rs',
                            'rs': '.rs',
                            'go': '.go',
                            'golang': '.go',
                            'zig': '.zig',
                            'd': '.d',
                            'nim': '.nim',
                            
                            # Mobile Development
                            'swift': '.swift',
                            'objc': '.m',
                            'objective-c': '.m',
                            'dart': '.dart',
                            
                            # Scripting
                            'bash': '.sh',
                            'sh': '.sh',
                            'shell': '.sh',
                            'zsh': '.zsh',
                            'fish': '.fish',
                            'powershell': '.ps1',
                            'ps1': '.ps1',
                            'batch': '.bat',
                            'bat': '.bat',
                            'cmd': '.cmd',
                            
                            # Data/Config
                            'sql': '.sql',
                            'yaml': '.yaml',
                            'yml': '.yml',
                            'toml': '.toml',
                            'xml': '.xml',
                            'json': '.json',
                            'ini': '.ini',
                            'cfg': '.cfg',
                            'conf': '.conf',
                            
                            # Other Popular Languages
                            'ruby': '.rb',
                            'rb': '.rb',
                            'perl': '.pl',
                            'pl': '.pl',
                            'r': '.R',
                            'julia': '.jl',
                            'jl': '.jl',
                            'lua': '.lua',
                            'tcl': '.tcl',
                            'haskell': '.hs',
                            'hs': '.hs',
                            'elm': '.elm',
                            'ocaml': '.ml',
                            'ml': '.ml',
                            'erlang': '.erl',
                            'erl': '.erl',
                            'elixir': '.ex',
                            'ex': '.ex',
                            
                            # Assembly
                            'assembly': '.asm',
                            'asm': '.asm',
                            'nasm': '.asm',
                            
                            # Specialized
                            'solidity': '.sol',
                            'sol': '.sol',
                            'verilog': '.v',
                            'v': '.v',
                            'vhdl': '.vhdl',
                            'latex': '.tex',
                            'tex': '.tex',
                            'markdown': '.md',
                            'md': '.md',
                            'rst': '.rst',
                            'restructuredtext': '.rst'
                        }
                        extension = extensions.get(language.lower(), '.txt')
                        
                        # Determine base filename from user request
                        base_filename = 'generated_file'
                        user_input_words = user_input.lower().split()
                        
                        # Look for common file name patterns for new files
                        if 'hello' in user_input_words and 'world' in user_input_words:
                            base_filename = 'hello_world'
                        elif 'index' in user_input_words:
                            base_filename = 'index'
                        elif 'main' in user_input_words:
                            base_filename = 'main'
                        elif 'app' in user_input_words:
                            base_filename = 'app'
                        elif 'server' in user_input_words:
                            base_filename = 'server'
                        elif 'client' in user_input_words:
                            base_filename = 'client'
                        elif 'test' in user_input_words:
                            base_filename = 'test'
                        elif 'example' in user_input_words:
                            base_filename = 'example'
                        
                        file_path = f"{base_filename}{extension}"
                    
                    # Create the new file
                    result = await self.code_generator.create_file(
                        file_path,
                        code_block['code'],
                        f"Generated from request: {user_input[:50]}...",
                        overwrite=False  # New files only, no overwriting
                    )
                    
                    if result['success']:
                        created_files.append({
                            'path': result['path'],
                            'size': result['size'],
                            'lines': result['lines']
                        })
                
                # Add file creation summary to response
                if created_files:
                    file_summary = "\n\n📁 **Files Created:**\n"
                    for file_info in created_files:
                        file_summary += f"✅ `{file_info['path']}` ({file_info['lines']} lines, {file_info['size']} bytes)\n"
                    
                    ai_response += file_summary
                    ai_response += "\n🎯 **Files are ready in your project directory!**"
            
            return ai_response
            
        except Exception as e:
            return f"❌ Error during code generation: {str(e)}\n\n{ai_response if 'ai_response' in locals() else ''}"
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown-formatted text"""
        import re
        
        print(f"[DEBUG] Extracting code blocks from text length: {len(text)}")
        
        # Pattern to match code blocks with language specification
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        print(f"[DEBUG] Found {len(matches)} markdown code blocks")
        
        code_blocks = []
        for language, code in matches:
            code_blocks.append({
                'language': language or 'text',
                'code': code.strip()
            })
            print(f"[DEBUG] Code block: language='{language}', code_length={len(code.strip())}")
        
        # Also look for inline code that might be complete files
        if not code_blocks:
            # Look for code patterns without markdown
            lines = text.split('\n')
            potential_code = []
            in_code_section = False
            
            for line in lines:
                # Common code indicators
                if any(indicator in line for indicator in ['def ', 'class ', 'import ', 'function', 'var ', 'const ', '#!/']):
                    in_code_section = True
                
                if in_code_section:
                    potential_code.append(line)
                
                # End of code section
                if line.strip() == '' and in_code_section and len(potential_code) > 3:
                    in_code_section = False
                    if potential_code:
                        code_blocks.append({
                            'language': 'python',  # Default assumption
                            'code': '\n'.join(potential_code).strip()
                        })
                        potential_code = []
        
        return code_blocks
    
    def _extract_file_paths(self, user_input: str) -> List[str]:
        """Extract file paths from user input"""
        import re
        
        file_paths = []
        
        # Look for explicit file paths
        patterns = [
            r'(?:save to|write to|create|in file|to file)\s+["\']?([^\s"\']+\.[a-zA-Z]+)["\']?',
            r'["\']([^\s"\']+\.[a-zA-Z]+)["\']',
            r'(\w+\.[a-zA-Z]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            file_paths.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in file_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        
        return unique_paths
    
    def _is_file_operation_request(self, user_input: str) -> bool:
        """Check if user input is requesting a simple file operation"""
        user_input_lower = user_input.lower()
        
        # File operation patterns
        file_operation_keywords = [
            'rename', 'move', 'delete', 'copy', 'duplicate', 
            'remove file', 'delete file', 'rename file',
            'move file', 'copy file', 'duplicate file'
        ]
        
        # Check for patterns like "rename X to Y" or "rename X.html to Y.html"
        rename_patterns = [
            r'rename\s+[\w.-]+\s+to\s+[\w.-]+',
            r'rename\s+[\w.-]+\.[\w]+\s+to\s+[\w.-]+\.[\w]+',
            r'change\s+[\w.-]+\s+to\s+[\w.-]+',
            r'move\s+[\w.-]+\s+to\s+[\w.-]+',
        ]
        
        # Check direct keywords
        if any(keyword in user_input_lower for keyword in file_operation_keywords):
            return True
        
        # Check rename patterns
        import re
        for pattern in rename_patterns:
            if re.search(pattern, user_input_lower):
                return True
        
        return False
    
    async def _handle_file_operations(self, user_input: str) -> str:
        """Handle simple file operations like rename, move, delete, etc."""
        if not self.code_generator:
            return "❌ File operations require a project directory. Please set up a project first."
        
        try:
            import os
            import re
            from pathlib import Path
            
            user_input_lower = user_input.lower()
            project_path = Path(self.code_generator.project_path)
            
            # Handle rename operations
            rename_match = re.search(r'rename\s+([\w.-]+(?:\.[\w]+)?)\s+to\s+([\w.-]+(?:\.[\w]+)?)', user_input_lower)
            if rename_match:
                old_name = rename_match.group(1)
                new_name = rename_match.group(2)
                
                # Find the file in project directory
                old_file_path = None
                for file_path in project_path.glob('**/*'):
                    if file_path.is_file() and file_path.name.lower() == old_name.lower():
                        old_file_path = file_path
                        break
                
                if not old_file_path:
                    return f"❌ File '{old_name}' not found in project directory."
                
                # Create new file path
                new_file_path = old_file_path.parent / new_name
                
                if new_file_path.exists():
                    return f"❌ File '{new_name}' already exists. Choose a different name."
                
                # Perform the rename
                old_file_path.rename(new_file_path)
                
                return f"✅ Successfully renamed `{old_name}` to `{new_name}`\n📁 Location: {new_file_path.relative_to(project_path)}"
            
            # Handle move operations
            move_match = re.search(r'move\s+([\w.-]+(?:\.[\w]+)?)\s+to\s+([\w/.-]+)', user_input_lower)
            if move_match:
                file_name = move_match.group(1)
                target_location = move_match.group(2)
                
                # Find the file
                old_file_path = None
                for file_path in project_path.glob('**/*'):
                    if file_path.is_file() and file_path.name.lower() == file_name.lower():
                        old_file_path = file_path
                        break
                
                if not old_file_path:
                    return f"❌ File '{file_name}' not found in project directory."
                
                # Create target directory if it doesn't exist
                if '/' in target_location or '\\' in target_location:
                    target_dir = project_path / target_location.replace('\\', '/')
                    target_dir.mkdir(parents=True, exist_ok=True)
                    new_file_path = target_dir / old_file_path.name
                else:
                    new_file_path = project_path / target_location
                
                if new_file_path.exists():
                    return f"❌ Target location already has a file with that name."
                
                # Perform the move
                old_file_path.rename(new_file_path)
                
                return f"✅ Successfully moved `{file_name}` to `{new_file_path.relative_to(project_path)}`"
            
            # Handle delete operations
            if any(keyword in user_input_lower for keyword in ['delete', 'remove']):
                delete_match = re.search(r'(?:delete|remove)\s+(?:file\s+)?([\w.-]+(?:\.[\w]+)?)', user_input_lower)
                if delete_match:
                    file_name = delete_match.group(1)
                    
                    # Find the file
                    file_to_delete = None
                    for file_path in project_path.glob('**/*'):
                        if file_path.is_file() and file_path.name.lower() == file_name.lower():
                            file_to_delete = file_path
                            break
                    
                    if not file_to_delete:
                        return f"❌ File '{file_name}' not found in project directory."
                    
                    # Perform the deletion
                    file_to_delete.unlink()
                    
                    return f"✅ Successfully deleted `{file_name}`"
            
            # Handle copy operations
            copy_match = re.search(r'(?:copy|duplicate)\s+([\w.-]+(?:\.[\w]+)?)\s+(?:to\s+|as\s+)?([\w.-]+(?:\.[\w]+)?)', user_input_lower)
            if copy_match:
                source_name = copy_match.group(1)
                target_name = copy_match.group(2)
                
                # Find source file
                source_file = None
                for file_path in project_path.glob('**/*'):
                    if file_path.is_file() and file_path.name.lower() == source_name.lower():
                        source_file = file_path
                        break
                
                if not source_file:
                    return f"❌ File '{source_name}' not found in project directory."
                
                target_file = source_file.parent / target_name
                
                if target_file.exists():
                    return f"❌ File '{target_name}' already exists."
                
                # Copy the file
                import shutil
                shutil.copy2(source_file, target_file)
                
                return f"✅ Successfully copied `{source_name}` to `{target_name}`"
            
            # If we get here, we couldn't parse the file operation
            return f"❌ I couldn't understand the file operation. Try:\n• `rename oldfile.txt to newfile.txt`\n• `move file.txt to folder/`\n• `delete file.txt`\n• `copy file.txt to backup.txt`"
            
        except Exception as e:
            return f"❌ Error during file operation: {str(e)}"
    
    async def create_file_directly(self, file_path: str, content: str, description: str = None) -> Dict[str, Any]:
        """Direct method to create files in the project"""
        if not self.code_generator:
            return {
                'success': False,
                'error': 'Code generator not available'
            }
        
        return await self.code_generator.create_file(
            file_path,
            content,
            description or f"File created directly: {file_path}"
        )
    
    async def _auto_switch_agent_if_needed(self, user_input: str, context: Dict[str, Any] = None):
        """Automatically switch agents based on user request and project context"""
        # We need access to the agent manager to switch agents
        # This will be injected by the main application
        if not hasattr(self, 'agent_manager') or not self.agent_manager:
            return
            
        user_input_lower = user_input.lower()
        current_agent_name = self.agent.name if self.agent else None
        
        # Define agent switching patterns
        agent_patterns = {
            'genesis-architect': [
                'design', 'architecture', 'plan', 'structure', 'organize',
                'blueprint', 'layout', 'system design', 'schema', 'framework',
                'how should i structure', 'what architecture', 'design pattern'
            ],
            'genesis-builder': [
                'build', 'implement', 'create', 'write code', 'develop',
                'generate', 'make', 'construct', 'code this', 'build this',
                'implement this', 'create a function', 'write a class'
            ],
            'genesis-tester': [
                'test', 'testing', 'unit test', 'debug', 'fix bug',
                'error', 'issue', 'problem', 'not working', 'broken',
                'write test', 'test coverage', 'quality assurance'
            ],
            'genesis-optimizer': [
                'optimize', 'performance', 'faster', 'improve', 'refactor',
                'clean up', 'better', 'efficient', 'speed up', 'memory',
                'benchmark', 'profiling', 'optimization'
            ],
            'genesis-deployer': [
                'deploy', 'deployment', 'production', 'server', 'hosting',
                'docker', 'container', 'kubernetes', 'aws', 'cloud',
                'publish', 'release', 'launch', 'go live'
            ]
        }
        
        # Find the best matching agent
        best_agent = None
        max_matches = 0
        
        for agent_name, patterns in agent_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in user_input_lower)
            if matches > max_matches:
                max_matches = matches
                best_agent = agent_name
        
        # Also consider Genesis phase context
        if context and self.genesis_engine:
            try:
                current_phase = await self.genesis_engine.get_current_phase()
                phase_agent_map = {
                    'design': 'genesis-architect',
                    'build': 'genesis-builder', 
                    'test': 'genesis-tester',
                    'deploy': 'genesis-deployer'
                }
                
                phase_suggested_agent = phase_agent_map.get(current_phase)
                if phase_suggested_agent and max_matches < 2:  # Only if no strong pattern match
                    best_agent = phase_suggested_agent
                    max_matches = 1
                    
            except Exception:
                pass  # Ignore if Genesis engine is not available
        
        # Switch agent if we found a better match and it's different from current
        if best_agent and best_agent != current_agent_name and max_matches > 0:
            try:
                # Get available agents
                available_agents = self.agent_manager.get_available_agents()
                agent_names = [agent.name for agent in available_agents]
                
                if best_agent in agent_names:
                    # Switch to the better agent
                    if await self.agent_manager.switch_agent(best_agent):
                        self.agent = self.agent_manager.current_agent
                        
                        # Show user that we switched agents
                        from rich.console import Console
                        console = Console()
                        
                        # Get agent descriptions for user context
                        agent_descriptions = {
                            'genesis-architect': 'system architecture and design',
                            'genesis-builder': 'code implementation and development',
                            'genesis-tester': 'testing and debugging',
                            'genesis-optimizer': 'performance optimization', 
                            'genesis-deployer': 'deployment and production'
                        }
                        
                        description = agent_descriptions.get(best_agent, 'specialized assistance')
                        
                        # Show switch notification with Genesis theme
                        console.print(f"\n[dim]🔄 Switched to [cyan]{best_agent}[/cyan] for {description}[/dim]")
                        
            except Exception as e:
                # Silently handle agent switching errors
                pass
    
    def set_agent_manager(self, agent_manager):
        """Set the agent manager for automatic switching"""
        self.agent_manager = agent_manager