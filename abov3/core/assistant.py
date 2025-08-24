"""
ABOV3 Genesis - Core Assistant
The main AI assistant that processes user requests and coordinates with other systems
"""

import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import json

from .ollama_client import OllamaClient

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
        self.default_model = "llama3:latest"
    
    async def process(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process user input and return response"""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Get model and system prompt from agent
            model = self.agent.model if self.agent else self.default_model
            system_prompt = self._build_system_prompt()
            
            # Check if Ollama is available
            if not await self.ollama_client.is_available():
                return self._get_fallback_response(user_input)
            
            # Prepare messages for chat
            messages = self._prepare_messages(user_input, system_prompt)
            
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