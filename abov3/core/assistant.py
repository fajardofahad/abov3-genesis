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
from .project_intelligence import ProjectIntelligence
from .app_generator import FullApplicationGenerator

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
        
        # Initialize code generator, project intelligence, and app generator if we have project context
        self.code_generator = None
        self.project_intelligence = None
        self.app_generator = None
        if project_context and 'project_path' in project_context:
            from pathlib import Path
            project_path = Path(project_context['project_path'])
            self.code_generator = CodeGenerator(project_path)
            self.project_intelligence = ProjectIntelligence(project_path)
            self.app_generator = FullApplicationGenerator(project_path)
    
    async def process(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process user input and return response"""
        try:
            # Update context if provided
            if context:
                self.project_context.update(context)
                
                # Initialize or update all generators if we have project path
                if 'project_path' in context and not self.code_generator:
                    from pathlib import Path
                    project_path = Path(context['project_path'])
                    self.code_generator = CodeGenerator(project_path)
                    self.project_intelligence = ProjectIntelligence(project_path)
                    self.app_generator = FullApplicationGenerator(project_path)
                    print(f"[DEBUG] Initialized all generators with path: {context['project_path']}")
                elif 'project_path' in context and self.code_generator:
                    # Update if project path changed
                    from pathlib import Path
                    new_path = Path(context['project_path'])
                    if new_path != self.code_generator.project_path:
                        self.code_generator = CodeGenerator(new_path)
                        self.project_intelligence = ProjectIntelligence(new_path)
                        self.app_generator = FullApplicationGenerator(new_path)
                        print(f"[DEBUG] Updated all generators with new path: {context['project_path']}")
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
            
            # Analyze project if we have project intelligence and it's needed
            if self.project_intelligence and self._should_analyze_project(user_input):
                print("[DEBUG] Running project analysis...")
                await self.project_intelligence.analyze_project()
            
            # Get model and system prompt from agent
            model = self.agent.model if self.agent else self.default_model
            system_prompt = self._build_system_prompt()
            
            # Check if Ollama is available
            if not await self.ollama_client.is_available():
                return self._get_fallback_response(user_input)
            
            # Prepare messages for chat
            messages = self._prepare_messages(user_input, system_prompt)
            
            # Check for different types of requests in priority order
            is_full_app_request = self._is_full_application_request(user_input)
            is_project_status = self._is_project_status_request(user_input)
            is_file_operation = self._is_file_operation_request(user_input)
            is_debug_request = self._is_debug_request(user_input)
            is_code_request = self._is_code_generation_request(user_input)
            
            print(f"[DEBUG] Full application request: {is_full_app_request}")
            print(f"[DEBUG] Project status request: {is_project_status}")
            print(f"[DEBUG] File operation request: {is_file_operation}")
            print(f"[DEBUG] Debug/error fixing request: {is_debug_request}")
            print(f"[DEBUG] Code generation request: {is_code_request}")
            
            if is_full_app_request:
                response_text = await self._handle_full_application_request(user_input, messages, model)
            elif is_project_status:
                response_text = await self._handle_project_status(user_input, messages, model)
            elif is_code_request:  # Code modification should come before file operations
                response_text = await self._handle_code_generation(user_input, messages, model)
            elif is_debug_request:
                response_text = await self._handle_debug_request(user_input, messages, model)
            elif is_file_operation:  # File operations last (for simple file ops like rename/delete)
                response_text = await self._handle_file_operations(user_input)
            else:
                # Get AI response
                response_text = await self._get_ai_response(model, messages)
            
            # Record interaction for learning if we have project intelligence
            if self.project_intelligence:
                action_type = 'full_app' if is_full_app_request else 'project_status' if is_project_status else 'file_op' if is_file_operation else 'debug' if is_debug_request else 'code_gen' if is_code_request else 'conversation'
                await self.project_intelligence.record_interaction(
                    user_input, response_text, 
                    action_taken=action_type
                )
            
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
    
    def _should_analyze_project(self, user_input: str) -> bool:
        """Determine if we should run project analysis based on user input"""
        analysis_triggers = [
            'update', 'modify', 'change', 'add', 'create', 'build', 'develop',
            'what is this project', 'explain this project', 'project status',
            'what does this do', 'how does this work', 'analyze', 'review'
        ]
        
        user_input_lower = user_input.lower()
        return any(trigger in user_input_lower for trigger in analysis_triggers)
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with context"""
        base_prompt = """You are ABOV3 Genesis, a code generation assistant.

RULES:
- Be extremely concise and direct
- Generate code immediately without explanations
- No greetings, no pleasantries, no "I'd be happy to help"
- When asked for code, respond ONLY with the code
- Use markdown code blocks with appropriate language tags
- Do not explain what the code does unless specifically asked
- Do not offer additional help or ask questions
- Just generate the requested code

Example:
User: make me an html hello world
Assistant: ```html
<!DOCTYPE html>
<html>
<head>
    <title>Hello World</title>
</head>
<body>
    <h1>Hello, World!</h1>
</body>
</html>
```

Nothing more, nothing less."""
        
        # Add agent-specific context
        if self.agent and hasattr(self.agent, 'system_prompt') and self.agent.system_prompt:
            base_prompt += f"\n\nSPECIALIZATION:\n{self.agent.system_prompt}"
        
        # Add project context from project intelligence or legacy context
        if self.project_intelligence:
            # Use intelligent project analysis
            project_context = self.project_intelligence.get_context_for_ai()
            if project_context.strip() != "PROJECT CONTEXT:\n- Name: " + str(self.project_intelligence.project_path.name):
                base_prompt += f"\n\n{project_context}"
        elif self.project_context:
            # Fallback to legacy project context
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
            from pathlib import Path
            project_path = Path(self.code_generator.project_path)
            
            # Check if project has existing files
            existing_files = []
            for file_path in project_path.rglob('*'):
                if file_path.is_file() and file_path.suffix in ['.html', '.css', '.js', '.py', '.md', '.txt']:
                    existing_files.append(file_path)
            
            # If no existing files OR this is clearly a "make/create" request, do new file creation
            is_creation_request = any(phrase in user_input.lower() for phrase in [
                'make me', 'create', 'build', 'generate', 'new website', 'new app'
            ])
            
            if not existing_files or is_creation_request:
                return await self._handle_new_file_creation(user_input, messages, model)
            else:
                # Has existing files and looks like a modification
                return await self._handle_file_modification(user_input, messages, model)
                
        except Exception as e:
            return f"❌ Error during code generation: {str(e)}"
    
    async def _handle_file_modification(self, user_input: str, messages: List[Dict[str, str]], model: str) -> str:
        """Handle modification of existing files like Claude does - simple and direct"""
        from pathlib import Path
        
        try:
            project_path = Path(self.code_generator.project_path)
            
            # Find all relevant files in project
            all_files = []
            for file_path in project_path.rglob('*'):
                if file_path.is_file() and file_path.suffix in ['.html', '.css', '.js', '.py', '.md', '.txt']:
                    all_files.append(file_path)
            
            if not all_files:
                return "❌ No files found in project to modify"
            
            # Read all files
            file_contents = {}
            for file_path in all_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_contents[str(file_path)] = content
                except:
                    continue
            
            if not file_contents:
                return "❌ Could not read any project files"
            
            # Create simple prompt for the AI like Claude does
            files_context = []
            for file_path, content in file_contents.items():
                file_name = Path(file_path).name
                files_context.append(f"**{file_name}:**\n```\n{content}\n```\n")
            
            prompt = f"""I need to modify files in this project. Here are the current files:

{chr(10).join(files_context[:5])}  

User request: {user_input}

Please provide the complete updated file content for each file that needs to be changed. Use this format:

**filename.ext**
```
complete updated content here
```

Only show files that actually need changes."""
            
            # Get AI response
            ai_messages = [{"role": "user", "content": prompt}]
            response = await self._get_ai_response(model, ai_messages)
            
            # Apply changes directly - parse the AI response for file blocks
            import re
            file_blocks = re.findall(r'\*\*([^*]+)\*\*\s*```[^\n]*\n(.*?)```', response, re.DOTALL)
            
            if not file_blocks:
                return response  # Just return AI response if no file blocks found
            
            modified_files = []
            for filename, new_content in file_blocks:
                filename = filename.strip()
                new_content = new_content.strip()
                
                # Find matching file
                target_file = None
                for file_path in all_files:
                    if file_path.name == filename:
                        target_file = file_path
                        break
                
                if target_file:
                    try:
                        with open(target_file, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        modified_files.append(filename)
                    except Exception as e:
                        print(f"[DEBUG] Failed to write {filename}: {e}")
            
            if modified_files:
                return f"✅ **Files updated successfully!**\n\n📝 Modified files:\n" + '\n'.join([f"• {f}" for f in modified_files]) + f"\n\n{response}"
            else:
                return response
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    
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
                
                # Return only file creation summary if files were created, otherwise return AI response
                if created_files:
                    file_summary = "📁 **Files Created:**\n"
                    for file_info in created_files:
                        file_summary += f"✅ `{file_info['path']}` ({file_info['lines']} lines, {file_info['size']} bytes)\n"
                    
                    file_summary += "\n🎯 **Files are ready in your project directory!**"
                    return file_summary  # Return only the file summary, not the verbose AI response
            
            # Only return AI response if no files were created
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
        # But make sure they're actually file operations, not content changes
        rename_patterns = [
            r'rename\s+[\w.-]+\.[\w]+\s+to\s+[\w.-]+\.[\w]+',  # Must have file extension
            r'rename\s+file\s+[\w.-]+\s+to\s+[\w.-]+',         # Must mention "file"
            r'move\s+[\w.-]+\.[\w]+\s+to\s+[\w.-]+',           # Must have file extension
            r'move\s+file\s+[\w.-]+\s+to\s+[\w.-]+',           # Must mention "file"
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
                    
                    # Check if this is a confirmation (user said yes/confirm/proceed etc)
                    if any(word in user_input_lower for word in ['yes', 'confirm', 'proceed', 'continue']):
                        # Perform the deletion
                        file_to_delete.unlink()
                        return f"✅ Successfully deleted `{file_name}`"
                    else:
                        # Ask for confirmation first
                        relative_path = file_to_delete.relative_to(project_path)
                        return f"⚠️  **DELETION CONFIRMATION REQUIRED**\n\n📁 File to delete: `{relative_path}`\n🗂️  Full path: `{file_to_delete}`\n\n**This action cannot be undone!**\n\n❓ Are you sure you want to delete this file?\n\n💭 Reply with **'yes delete {file_name}'** or **'confirm delete {file_name}'** to proceed.\n💭 Reply with anything else to cancel."
            
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
    
    def _is_debug_request(self, user_input: str) -> bool:
        """Check if user input is requesting debugging or error fixing"""
        user_input_lower = user_input.lower()
        
        # Debug and error-related keywords
        debug_keywords = [
            'debug', 'fix', 'error', 'bug', 'issue', 'problem', 'broken',
            'not working', 'doesnt work', "doesn't work", 'failing', 'crash',
            'exception', 'traceback', 'syntax error', 'runtime error',
            'find error', 'find bug', 'whats wrong', "what's wrong",
            'analyze', 'check code', 'review code', 'troubleshoot',
            'diagnose', 'investigate', 'examine', 'lint', 'validate'
        ]
        
        # Error patterns - common error descriptions
        error_patterns = [
            r'getting.*error',
            r'throws.*error',
            r'returns.*error',
            r'shows.*error', 
            r'error.*when',
            r'error.*trying',
            r'cant.*work',
            r"can't.*work",
            r'wont.*work',
            r"won't.*work",
            r'failing.*to',
            r'unable.*to',
            r'problem.*with',
            r'issue.*with',
            r'broken.*code',
            r'code.*broken'
        ]
        
        # Check direct keywords
        if any(keyword in user_input_lower for keyword in debug_keywords):
            return True
        
        # Check error patterns
        import re
        for pattern in error_patterns:
            if re.search(pattern, user_input_lower):
                return True
        
        return False
    
    async def _handle_debug_request(self, user_input: str, messages: List[Dict[str, str]], model: str) -> str:
        """Handle debugging and error fixing requests"""
        if not self.code_generator:
            return "❌ Debugging requires a project directory. Please set up a project first."
        
        try:
            from pathlib import Path
            import os
            import re
            
            project_path = Path(self.code_generator.project_path)
            user_input_lower = user_input.lower()
            
            print(f"[DEBUG] Starting debug analysis for request: {user_input[:50]}...")
            
            # Step 1: Find relevant files to analyze
            relevant_files = []
            
            # Look for specific files mentioned in the user input
            mentioned_files = self._extract_file_paths(user_input)
            if mentioned_files:
                for file_name in mentioned_files:
                    for file_path in project_path.glob('**/*'):
                        if file_path.is_file() and file_path.name.lower() == file_name.lower():
                            relevant_files.append(file_path)
                            break
            
            # If no specific files mentioned, find files based on context
            if not relevant_files:
                # Look for common file types that might have issues
                file_patterns = []
                
                if any(word in user_input_lower for word in ['python', 'py', 'script']):
                    file_patterns.extend(['*.py'])
                elif any(word in user_input_lower for word in ['javascript', 'js', 'node']):
                    file_patterns.extend(['*.js', '*.jsx'])
                elif any(word in user_input_lower for word in ['web', 'html', 'website', 'page']):
                    file_patterns.extend(['*.html', '*.css', '*.js'])
                elif any(word in user_input_lower for word in ['java']):
                    file_patterns.extend(['*.java'])
                elif any(word in user_input_lower for word in ['cpp', 'c++', 'c']):
                    file_patterns.extend(['*.cpp', '*.c', '*.h'])
                else:
                    # Default: look for common programming files
                    file_patterns = ['*.py', '*.js', '*.html', '*.css', '*.java', '*.cpp', '*.c']
                
                # Find files matching patterns
                for pattern in file_patterns:
                    try:
                        for file_path in project_path.glob(pattern):
                            if file_path.is_file():
                                relevant_files.append(file_path)
                                if len(relevant_files) >= 5:  # Limit to 5 files max
                                    break
                    except:
                        continue
                    if len(relevant_files) >= 5:
                        break
            
            if not relevant_files:
                return "❌ No code files found to debug. Please specify which file has the issue or make sure your project contains code files."
            
            print(f"[DEBUG] Found {len(relevant_files)} files for debug analysis")
            
            # Step 2: Read and analyze file contents
            file_contents = {}
            for file_path in relevant_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_contents[file_path.name] = {
                            'path': file_path,
                            'content': content,
                            'extension': file_path.suffix
                        }
                        print(f"[DEBUG] Read file for analysis: {file_path.name} ({len(content)} chars)")
                except Exception as e:
                    print(f"[DEBUG] Could not read {file_path}: {e}")
            
            if not file_contents:
                return "❌ Could not read any files for debugging analysis."
            
            # Step 3: Run basic static analysis
            analysis_results = await self._perform_static_analysis(file_contents)
            
            # Step 4: Create enhanced debugging prompt for AI
            debug_prompt = f"""
I need help debugging code. Here are the files and any detected issues:

USER REQUEST: {user_input}

CODE FILES:
{chr(10).join([f"**{filename}** ({info['extension']}):{chr(10)}```{info['extension'][1:] if info['extension'] else 'text'}{chr(10)}{info['content']}{chr(10)}```{chr(10)}" for filename, info in file_contents.items()])}

STATIC ANALYSIS RESULTS:
{analysis_results}

Please analyze the code and:
1. Identify any syntax errors, logical issues, or potential bugs
2. Explain what might be causing the problem described by the user
3. Provide specific fixes with corrected code
4. Suggest improvements or best practices
5. If error messages were provided, explain what they mean and how to fix them

Format your response with clear explanations and provide complete corrected code blocks where needed.
"""
            
            # Step 5: Get AI debugging response
            debug_messages = messages[:-1] + [{"role": "user", "content": debug_prompt}]
            ai_response = await self._get_ai_response(model, debug_messages)
            
            # Step 6: Extract any code fixes from AI response
            code_blocks = self._extract_code_blocks(ai_response)
            fixed_files = []
            
            if code_blocks:
                print(f"[DEBUG] AI provided {len(code_blocks)} code fixes")
                
                # Ask user if they want to apply fixes (in a real implementation)
                # For now, we'll just show what would be fixed
                for i, code_block in enumerate(code_blocks):
                    # Try to match code block to appropriate file
                    target_file = None
                    language = code_block.get('language', '').lower()
                    
                    # Smart file matching based on language and content
                    if language in ['python', 'py']:
                        target_file = next((info for name, info in file_contents.items() if name.endswith('.py')), None)
                    elif language in ['javascript', 'js']:
                        target_file = next((info for name, info in file_contents.items() if name.endswith(('.js', '.jsx'))), None)
                    elif language in ['html', 'htm']:
                        target_file = next((info for name, info in file_contents.items() if name.endswith(('.html', '.htm'))), None)
                    elif language == 'css':
                        target_file = next((info for name, info in file_contents.items() if name.endswith('.css')), None)
                    elif language in ['java']:
                        target_file = next((info for name, info in file_contents.items() if name.endswith('.java')), None)
                    elif language in ['cpp', 'c++', 'c']:
                        target_file = next((info for name, info in file_contents.items() if name.endswith(('.cpp', '.c', '.h'))), None)
                    
                    # If no specific match, try to match by content similarity or use first file
                    if not target_file and file_contents:
                        target_file = list(file_contents.values())[0]
                    
                    if target_file:
                        # Check if the code actually fixes something (basic heuristic)
                        if len(code_block['code']) > len(target_file['content']) * 0.3:  # Substantial change
                            try:
                                # Write the fix back to the file
                                with open(target_file['path'], 'w', encoding='utf-8') as f:
                                    f.write(code_block['code'])
                                
                                fixed_files.append({
                                    'path': target_file['path'].name,
                                    'full_path': str(target_file['path']),
                                    'size': len(code_block['code']),
                                    'lines': code_block['code'].count('\n') + 1
                                })
                                print(f"[DEBUG] Applied fix to: {target_file['path'].name}")
                            except Exception as e:
                                print(f"[DEBUG] Failed to apply fix to {target_file['path'].name}: {e}")
            
            # Step 7: Prepare final response
            if fixed_files:
                fix_summary = "\n\n🔧 **Files Fixed:**\n"
                for file_info in fixed_files:
                    fix_summary += f"✅ `{file_info['path']}` ({file_info['lines']} lines, {file_info['size']} bytes)\n"
                
                ai_response += fix_summary
                ai_response += "\n🎯 **Debug fixes have been applied to your files!**"
            else:
                ai_response += "\n\n💡 **Analysis complete!** Review the suggestions above to fix the issues."
            
            return ai_response
            
        except Exception as e:
            return f"❌ Error during debugging analysis: {str(e)}"
    
    async def _perform_static_analysis(self, file_contents: Dict[str, Dict]) -> str:
        """Perform basic static analysis on code files"""
        results = []
        
        for filename, info in file_contents.items():
            content = info['content']
            extension = info.get('extension', '')
            issues = []
            
            # Python-specific analysis
            if extension == '.py':
                # Check for common Python issues
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    # Indentation issues (basic check)
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t') and ':' in line:
                        if i < len(lines) and lines[i].strip() and not lines[i].startswith((' ', '\t')):
                            issues.append(f"Line {i+1}: Possible indentation issue after '{line.strip()}'")
                    
                    # Common syntax issues
                    if '=' in line and '==' not in line and '!=' not in line and '>=' not in line and '<=' not in line:
                        if 'if ' in line or 'while ' in line or 'elif ' in line:
                            issues.append(f"Line {i}: Possible assignment instead of comparison in condition")
                    
                    # Missing imports
                    if any(func in line for func in ['os.', 'sys.', 'json.', 'datetime.']) and not any(imp in content for imp in ['import os', 'import sys', 'import json', 'import datetime']):
                        issues.append(f"Line {i}: Using module functions without import")
            
            # JavaScript-specific analysis  
            elif extension in ['.js', '.jsx']:
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    # Missing semicolons (basic check)
                    if line.strip().endswith((')', ']', '}')) and not line.strip().endswith((';', '{', ',')):
                        if not any(keyword in line for keyword in ['if', 'while', 'for', 'function', 'class']):
                            issues.append(f"Line {i}: Possibly missing semicolon")
                    
                    # Undefined variables (very basic)
                    if '=' in line and not any(keyword in line for keyword in ['var ', 'let ', 'const ', 'function ']):
                        var_name = line.split('=')[0].strip()
                        if var_name and not any(f'{var_name} ' in prev_line for prev_line in content.split('\n')[:i-1]):
                            issues.append(f"Line {i}: '{var_name}' might be used without declaration")
            
            # HTML-specific analysis
            elif extension in ['.html', '.htm']:
                # Basic HTML validation
                if '<html' in content and '</html>' not in content:
                    issues.append("Missing closing </html> tag")
                if '<body' in content and '</body>' not in content:
                    issues.append("Missing closing </body> tag")
                if '<head' in content and '</head>' not in content:
                    issues.append("Missing closing </head> tag")
            
            # CSS-specific analysis
            elif extension == '.css':
                # Basic CSS validation
                if content.count('{') != content.count('}'):
                    issues.append("Mismatched curly braces - possible unclosed CSS rule")
            
            if issues:
                results.append(f"**{filename}:**\n" + '\n'.join(f"- {issue}" for issue in issues))
            else:
                results.append(f"**{filename}:** No obvious issues detected")
        
        return '\n\n'.join(results) if results else "No static analysis issues detected."
    
    def _is_project_status_request(self, user_input: str) -> bool:
        """Check if user is asking about project status or understanding"""
        user_input_lower = user_input.lower()
        
        status_keywords = [
            'what is this project', 'explain this project', 'project status',
            'what does this do', 'how does this work', 'what am i working on',
            'project summary', 'tell me about this project', 'analyze project',
            'review project', 'project overview', 'what\'s this project about',
            'understand this project', 'project info', 'project details',
            'what language is this', 'what framework', 'what type of project'
        ]
        
        return any(keyword in user_input_lower for keyword in status_keywords)
    
    async def _handle_project_status(self, user_input: str, messages: List[Dict[str, str]], model: str) -> str:
        """Handle project status and analysis requests"""
        if not self.project_intelligence:
            return "❌ Project analysis requires a project directory. Please set up a project first."
        
        try:
            print("[DEBUG] Handling project status request")
            
            # Force analysis if requested
            await self.project_intelligence.analyze_project(force_reanalysis=True)
            
            # Get project summary
            summary = self.project_intelligence.get_project_summary()
            knowledge = self.project_intelligence.project_knowledge
            
            # Create detailed status response
            status_parts = [
                "🔍 **Project Analysis Complete**",
                f"📊 **Summary:** {summary}",
                ""
            ]
            
            # Add detailed information
            if knowledge.get('confidence_score', 0) >= 0.5:
                if knowledge.get('structure'):
                    structure = knowledge['structure']
                    status_parts.extend([
                        "📁 **Project Structure:**",
                        f"- Total files: {structure.get('total_files', 0)}",
                        f"- Directories: {len(structure.get('directories', []))}",
                        f"- Project size: {structure.get('size_bytes', 0)} bytes",
                        ""
                    ])
                
                if knowledge.get('languages'):
                    status_parts.append("💻 **Languages Used:**")
                    for lang, info in knowledge['languages'].items():
                        status_parts.append(f"- {lang.title()}: {info['files']} files ({info['percentage']:.1f}%)")
                    status_parts.append("")
                
                if knowledge.get('frameworks'):
                    frameworks = ', '.join(knowledge['frameworks'])
                    status_parts.extend([
                        "🛠️ **Frameworks & Libraries:**",
                        f"- {frameworks}",
                        ""
                    ])
                
                if knowledge.get('entry_points'):
                    entry_points = ', '.join(knowledge['entry_points'])
                    status_parts.extend([
                        "🚀 **Entry Points:**",
                        f"- {entry_points}",
                        ""
                    ])
                
                if knowledge.get('key_files'):
                    key_files = ', '.join(knowledge['key_files'])
                    status_parts.extend([
                        "📄 **Key Files:**",
                        f"- {key_files}",
                        ""
                    ])
                
                # Add recent work context
                interactions = self.project_intelligence.interactions
                if interactions:
                    status_parts.extend([
                        "📝 **Recent Activity:**",
                    ])
                    for interaction in interactions[-3:]:  # Last 3 interactions
                        timestamp = datetime.fromisoformat(interaction['timestamp']).strftime("%H:%M")
                        request = interaction['user_request'][:60] + '...' if len(interaction['user_request']) > 60 else interaction['user_request']
                        status_parts.append(f"- [{timestamp}] {request}")
                    status_parts.append("")
                
                confidence_emoji = "🔥" if knowledge['confidence_score'] >= 0.8 else "✅" if knowledge['confidence_score'] >= 0.6 else "⚠️"
                status_parts.append(f"{confidence_emoji} **Analysis Confidence:** {knowledge['confidence_score']:.0%}")
            else:
                status_parts.extend([
                    "⚠️ **Limited Information Available**",
                    "The project analysis is still building understanding of your codebase.",
                    "Continue working with the project to improve analysis accuracy."
                ])
            
            # Enhanced AI response with project context
            if 'what' in user_input.lower() or 'how' in user_input.lower() or 'explain' in user_input.lower():
                # User wants detailed explanation, enhance with AI analysis
                enhanced_prompt = f"""
Based on the project analysis, provide a detailed explanation of this project:

{self.project_intelligence.get_context_for_ai()}

USER QUESTION: {user_input}

Please provide a comprehensive explanation of what this project does, how it works, and its purpose. Use the analysis data above and be specific about the technologies and structure.
"""
                enhanced_messages = messages[:-1] + [{"role": "user", "content": enhanced_prompt}]
                ai_explanation = await self._get_ai_response(model, enhanced_messages)
                
                return '\n'.join(status_parts) + f"\n\n🤖 **AI Analysis:**\n{ai_explanation}"
            else:
                return '\n'.join(status_parts)
            
        except Exception as e:
            return f"❌ Error during project analysis: {str(e)}"
    
    def _is_full_application_request(self, user_input: str) -> bool:
        """Check if user is requesting a complete application to be built"""
        user_input_lower = user_input.lower()
        
        # Full application keywords
        full_app_keywords = [
            'make me a website', 'create a website', 'build a website',
            'make me an app', 'create an app', 'build an app',
            'make me a mobile app', 'create a mobile app', 'build a mobile app',
            'create a full', 'build a full', 'make a full',
            'complete website', 'complete application', 'complete app',
            'entire website', 'entire application', 'entire app',
            'full stack', 'end to end', 'production ready',
            'from scratch', 'ground up',
            'yes i want it all', 'give me all', 'add everything', 'include everything',
            'full featured', 'with all features', 'complete solution'
        ]
        
        # Business/domain specific requests that typically need full apps
        business_app_patterns = [
            'coffee shop website', 'restaurant website', 'e-commerce', 'online store',
            'bobba tea shop', 'boba tea shop', 'tea shop website', 'cafe website',
            'bakery website', 'food website', 'drinks website', 'beverage website',
            'portfolio website', 'business website', 'company website',
            'blog website', 'news website', 'social media app',
            'todo app', 'chat app', 'calendar app', 'note taking app',
            'inventory system', 'booking system', 'reservation system'
        ]
        
        # Check for explicit full application requests
        if any(keyword in user_input_lower for keyword in full_app_keywords):
            return True
        
        # Check for business/domain patterns that typically need full apps
        if any(pattern in user_input_lower for pattern in business_app_patterns):
            return True
        
        # Check for complex feature combinations that suggest full app
        feature_count = 0
        features = [
            'with login', 'with authentication', 'with user accounts',
            'with cart', 'with shopping', 'with payment',
            'with database', 'with admin', 'with dashboard',
            'with api', 'with backend', 'with search',
            'with comments', 'with reviews', 'with ratings'
        ]
        
        for feature in features:
            if feature in user_input_lower:
                feature_count += 1
        
        # If requesting multiple complex features, likely needs full app
        if feature_count >= 2:
            return True
        
        return False
    
    async def _handle_full_application_request(self, user_input: str, messages: List[Dict[str, str]], model: str) -> str:
        """Handle full application generation requests"""
        if not self.app_generator:
            return "❌ Full application generation requires a project directory. Please set up a project first."
        
        try:
            print(f"[DEBUG] Starting full application generation for: {user_input[:100]}...")
            
            # Extract preferences from user input (could be enhanced with AI analysis)
            preferences = await self._extract_app_preferences(user_input, messages, model)
            
            # Generate the complete application
            generation_result = await self.app_generator.generate_full_application(user_input, preferences)
            
            if generation_result.get('success'):
                # Prepare success response
                analysis = generation_result.get('analysis', {})
                architecture = generation_result.get('architecture', {})
                files = generation_result.get('generated_files', [])
                setup_result = generation_result.get('setup_result', {})
                next_steps = generation_result.get('next_steps', [])
                
                response_parts = [
                    "🚀 **Complete Application Generated Successfully!**",
                    "",
                    f"📊 **Application Analysis:**",
                    f"- Type: {analysis.get('app_type', 'Unknown').replace('_', ' ').title()}",
                    f"- Platform: {analysis.get('target_platform', 'Unknown').title()}",
                    f"- Complexity: {analysis.get('complexity', 'Unknown').title()}",
                    f"- Tech Stack: {architecture.get('tech_stack', 'Unknown').replace('_', ' ').title()}",
                    ""
                ]
                
                # Add features
                if analysis.get('features'):
                    response_parts.extend([
                        "✨ **Features Implemented:**",
                        *[f"- {feature.replace('_', ' ').title()}" for feature in analysis['features']],
                        ""
                    ])
                
                # Add generated components
                if architecture.get('pages'):
                    response_parts.extend([
                        "📄 **Pages Created:**",
                        *[f"- {page.replace('_', ' ').replace('-', ' ').title()}" for page in architecture['pages']],
                        ""
                    ])
                
                if architecture.get('components'):
                    response_parts.extend([
                        "🧩 **Components Generated:**",
                        *[f"- {component}" for component in architecture['components'][:5]],  # Show first 5
                        f"- ... and {len(architecture['components']) - 5} more" if len(architecture['components']) > 5 else "",
                        ""
                    ])
                
                # Add file summary
                response_parts.extend([
                    f"📁 **Files Generated:** {len(files)} files",
                    ""
                ])
                
                # Add environment setup results
                if setup_result:
                    response_parts.append("🔧 **Environment Setup:**")
                    
                    # Show installation logs
                    if setup_result.get('installation_logs'):
                        for log in setup_result['installation_logs'][:5]:  # Show first 5 logs
                            response_parts.append(f"{log}")
                        if len(setup_result['installation_logs']) > 5:
                            response_parts.append(f"... and {len(setup_result['installation_logs']) - 5} more setup steps")
                    
                    # Show setup commands
                    if setup_result.get('setup_commands'):
                        response_parts.extend([
                            "",
                            "🚀 **Ready to Run:**"
                        ])
                        for cmd in setup_result['setup_commands']:
                            response_parts.append(f"```bash\n{cmd}\n```")
                    
                    # Show errors if any
                    if setup_result.get('errors'):
                        response_parts.extend([
                            "",
                            "⚠️ **Setup Issues:**"
                        ])
                        for error in setup_result['errors'][:3]:  # Show first 3 errors
                            response_parts.append(f"- {error}")
                    
                    response_parts.append("")
                
                # Add deployment info
                deployment = generation_result.get('deployment', {})
                if deployment.get('instructions'):
                    response_parts.extend([
                        "🚀 **Quick Start:**",
                        *[f"- {instruction}" for instruction in deployment['instructions'][:3]],
                        ""
                    ])
                
                # Add next steps
                if next_steps:
                    response_parts.extend([
                        "📋 **Next Steps:**",
                        *[f"- {step}" for step in next_steps[:5]],
                        ""
                    ])
                
                response_parts.extend([
                    "📚 **Documentation:** Check README.md for complete setup instructions",
                    "🎯 **Your production-ready application is ready to deploy!**"
                ])
                
                return '\n'.join(response_parts)
            
            else:
                error = generation_result.get('error', 'Unknown error occurred')
                return f"❌ **Application Generation Failed**\n\nError: {error}\n\nPlease try again with a more specific request or check your project setup."
            
        except Exception as e:
            return f"❌ **Error during application generation:** {str(e)}\n\nPlease try again or provide more details about the application you want to create."
    
    async def _extract_app_preferences(self, user_input: str, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Extract user preferences for application generation using AI"""
        try:
            # Create prompt to extract preferences
            preference_prompt = f"""
Analyze this user request for building an application and extract their preferences:

USER REQUEST: {user_input}

Extract and return ONLY the following information in JSON format:

{{
    "tech_stack": "preferred technology (react_node_mongodb, html_css_js, flutter, etc.)",
    "features": ["list", "of", "requested", "features"],
    "style_preferences": "visual style preferences if mentioned",
    "target_audience": "who is this for",
    "complexity_preference": "simple/medium/advanced",
    "deployment_preference": "where they want to deploy if mentioned"
}}

Focus only on what is explicitly mentioned or clearly implied. If something is not mentioned, use appropriate defaults.
"""
            
            # Get AI analysis
            preference_messages = [{"role": "user", "content": preference_prompt}]
            ai_response = await self._get_ai_response(model, preference_messages)
            
            # Try to parse JSON response
            try:
                import json
                import re
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    preferences = json.loads(json_match.group(0))
                    return preferences
                else:
                    return {}
            except:
                return {}
                
        except Exception as e:
            print(f"[DEBUG] Error extracting preferences: {e}")
            return {}
    
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