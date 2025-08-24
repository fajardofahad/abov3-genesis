"""
ABOV3 Genesis Agent Commands
Command handlers for agent management
"""

from typing import List, Dict, Any, Optional
from .manager import AgentManager

class AgentCommandHandler:
    """
    Handles agent-related commands in ABOV3 Genesis
    """
    
    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
    
    async def handle_agent_command(self, command_parts: List[str]) -> str:
        """Handle agent commands"""
        if not command_parts:
            return self._show_current_agent()
        
        subcommand = command_parts[0].lower()
        
        if subcommand == 'list':
            return self._list_agents()
        elif subcommand == 'switch' and len(command_parts) > 1:
            return await self._switch_agent(command_parts[1])
        elif subcommand == 'create':
            return await self._create_agent_interactive()
        elif subcommand == 'delete' and len(command_parts) > 1:
            return await self._delete_agent(command_parts[1])
        elif subcommand == 'info' and len(command_parts) > 1:
            return self._show_agent_info(command_parts[1])
        elif subcommand == 'edit' and len(command_parts) > 1:
            return await self._edit_agent_interactive(command_parts[1])
        elif subcommand == 'reset':
            return await self._reset_agents()
        elif subcommand == 'stats':
            return self._show_agent_stats()
        elif subcommand == 'help':
            return self._show_agent_help()
        else:
            return f"Unknown agent command: {subcommand}. Type '/agents help' for available commands."
    
    def _show_current_agent(self) -> str:
        """Show current agent information"""
        if not self.agent_manager.current_agent:
            return "No agent is currently active."
        
        agent = self.agent_manager.current_agent
        return f"""🤖 **Current Agent**: {agent.name}
📝 **Description**: {agent.description}  
🧠 **Model**: {agent.model}
📊 **Usage Count**: {agent.usage_count}

Type '/agents switch <name>' to switch to a different agent."""
    
    def _list_agents(self) -> str:
        """List all available agents"""
        agents = self.agent_manager.get_available_agents()
        
        if not agents:
            return "No agents available. Creating default Genesis agents..."
        
        current_name = self.agent_manager.current_agent.name if self.agent_manager.current_agent else None
        
        result = "🤖 **Available Agents**:\n\n"
        
        # Group Genesis agents separately
        genesis_agents = self.agent_manager.get_genesis_agents()
        other_agents = [a for a in agents if a not in genesis_agents]
        
        if genesis_agents:
            result += "**Genesis Agents** (Specialized for idea-to-reality workflow):\n"
            for agent in genesis_agents:
                current_mark = "→ " if agent.name == current_name else "  "
                result += f"{current_mark}**{agent.name}**: {agent.description} `[{agent.model}]`\n"
        
        if other_agents:
            result += "\n**Custom Agents**:\n"
            for agent in other_agents:
                current_mark = "→ " if agent.name == current_name else "  "
                result += f"{current_mark}**{agent.name}**: {agent.description} `[{agent.model}]`\n"
        
        result += "\nType '/agents switch <name>' to switch agents."
        return result
    
    async def _switch_agent(self, agent_name: str) -> str:
        """Switch to a different agent"""
        if not self.agent_manager.has_agent(agent_name):
            available = [a.name for a in self.agent_manager.get_available_agents()]
            return f"Agent '{agent_name}' not found. Available agents: {', '.join(available)}"
        
        if await self.agent_manager.switch_agent(agent_name):
            agent = self.agent_manager.get_agent(agent_name)
            return f"✅ Switched to agent: **{agent_name}**\n📝 {agent.description}"
        else:
            return f"❌ Failed to switch to agent: {agent_name}"
    
    async def _create_agent_interactive(self) -> str:
        """Interactive agent creation"""
        return """🛠️ **Create New Agent**

To create a custom agent, you'll need:

1. **Name**: Unique identifier (e.g., 'my-specialist')  
2. **Model**: Ollama model to use (e.g., 'llama3:latest')
3. **Description**: Brief description of the agent's purpose
4. **System Prompt**: Detailed instructions for the agent's behavior

**Example**:
```
/agents create my-reviewer llama3:latest "Code review specialist" "You are a code review expert who focuses on finding bugs, security issues, and improvement opportunities. Always provide constructive feedback with specific suggestions."
```

**Available Models**: {models}

**Genesis Agents** are already optimized for the idea-to-reality workflow. Consider using them first!""".format(
            models=", ".join(self.agent_manager.get_recommended_models())
        )
    
    async def _delete_agent(self, agent_name: str) -> str:
        """Delete an agent"""
        if not self.agent_manager.has_agent(agent_name):
            return f"Agent '{agent_name}' not found."
        
        # Check if it's a Genesis agent
        genesis_agents = [a.name for a in self.agent_manager.get_genesis_agents()]
        if agent_name in genesis_agents:
            return f"Cannot delete Genesis agent '{agent_name}'. Use '/agents reset' to restore defaults."
        
        if await self.agent_manager.delete_agent(agent_name):
            return f"✅ Deleted agent: {agent_name}"
        else:
            return f"❌ Failed to delete agent: {agent_name}"
    
    def _show_agent_info(self, agent_name: str) -> str:
        """Show detailed agent information"""
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found."
        
        is_current = (self.agent_manager.current_agent and 
                     self.agent_manager.current_agent.name == agent_name)
        current_mark = " (Current)" if is_current else ""
        
        return f"""🤖 **Agent Details**: {agent.name}{current_mark}

📝 **Description**: {agent.description}
🧠 **Model**: {agent.model}  
📅 **Created**: {agent.created}
📊 **Usage Count**: {agent.usage_count}

**System Prompt**:
```
{agent.system_prompt[:300]}{'...' if len(agent.system_prompt) > 300 else ''}
```

Type '/agents switch {agent.name}' to use this agent."""
    
    async def _edit_agent_interactive(self, agent_name: str) -> str:
        """Interactive agent editing"""
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found."
        
        return f"""🛠️ **Edit Agent**: {agent_name}

Current configuration:
- **Model**: {agent.model}
- **Description**: {agent.description}

To edit an agent, use:
```
/agents update {agent_name} <field> <new_value>
```

Available fields:
- `model`: Change the Ollama model
- `description`: Update the description  
- `system_prompt`: Modify the system prompt

**Note**: Genesis agents have optimized prompts. Edit carefully!"""
    
    async def _reset_agents(self) -> str:
        """Reset to default Genesis agents"""
        await self.agent_manager.reset_to_defaults()
        return """✅ **Agents Reset Successfully**

Restored default Genesis agents:
- 🏗️ **genesis-architect**: Transforms ideas into system architecture
- 🔨 **genesis-builder**: Builds complete applications from scratch  
- 🎨 **genesis-designer**: Creates beautiful UIs from descriptions
- ⚡ **genesis-optimizer**: Optimizes and perfects existing code
- 🚀 **genesis-deployer**: Handles deployment and production readiness

Current agent: **genesis-architect**"""
    
    def _show_agent_stats(self) -> str:
        """Show agent statistics"""
        stats = self.agent_manager.get_agent_stats()
        
        result = f"""📊 **Agent Statistics**

📈 **Overview**:
- Total Agents: {stats['total_agents']}
- Genesis Agents: {stats['genesis_agents']}  
- Current Agent: {stats['current_agent']}
- Most Used: {stats.get('most_used', 'None')}

📋 **Agent Usage**:"""
        
        # Sort agents by usage
        agents = sorted(stats['agents'], key=lambda a: a['usage_count'], reverse=True)
        
        for agent in agents:
            result += f"\n• **{agent['name']}**: {agent['usage_count']} uses `[{agent['model']}]`"
        
        return result
    
    def _show_agent_help(self) -> str:
        """Show agent command help"""
        return """🤖 **Agent Commands Help**

**Basic Commands**:
- `/agents` - Show current agent
- `/agents list` - List all available agents
- `/agents switch <name>` - Switch to an agent
- `/agents info <name>` - Show agent details

**Management Commands**:
- `/agents create` - Create new agent (interactive)
- `/agents delete <name>` - Delete custom agent
- `/agents edit <name>` - Edit agent (interactive)
- `/agents reset` - Reset to Genesis defaults

**Information Commands**:
- `/agents stats` - Show usage statistics
- `/agents help` - Show this help

**Genesis Agents** (Pre-configured specialists):
- `genesis-architect` - System architecture design
- `genesis-builder` - Code implementation  
- `genesis-designer` - UI/UX creation
- `genesis-optimizer` - Code optimization
- `genesis-deployer` - Deployment & production

**Examples**:
```
/agents switch genesis-builder
/agents info genesis-architect
/agents create my-tester llama3:latest "Testing specialist" "You create comprehensive tests"
```"""
    
    async def create_agent_from_command(self, args: List[str]) -> str:
        """Create agent from command line arguments"""
        if len(args) < 4:
            return "Usage: /agents create <name> <model> <description> <system_prompt>"
        
        name = args[0]
        model = args[1]
        description = args[2]
        system_prompt = ' '.join(args[3:])
        
        # Validate configuration
        config = {
            'name': name,
            'model': model,
            'description': description,
            'system_prompt': system_prompt
        }
        
        is_valid, message = await self.agent_manager.validate_agent_config(config)
        if not is_valid:
            return f"❌ Invalid agent configuration: {message}"
        
        # Create agent
        if await self.agent_manager.create_agent(name, model, description, system_prompt):
            return f"✅ Created agent: **{name}**\n📝 {description}\nType '/agents switch {name}' to use it."
        else:
            return f"❌ Failed to create agent: {name}"
    
    async def update_agent_from_command(self, args: List[str]) -> str:
        """Update agent from command line arguments"""
        if len(args) < 3:
            return "Usage: /agents update <name> <field> <new_value>"
        
        name = args[0]
        field = args[1].lower()
        new_value = ' '.join(args[2:])
        
        if not self.agent_manager.has_agent(name):
            return f"Agent '{name}' not found."
        
        valid_fields = ['model', 'description', 'system_prompt']
        if field not in valid_fields:
            return f"Invalid field '{field}'. Valid fields: {', '.join(valid_fields)}"
        
        # Update agent
        kwargs = {field: new_value}
        if await self.agent_manager.update_agent(name, **kwargs):
            return f"✅ Updated {field} for agent: {name}"
        else:
            return f"❌ Failed to update agent: {name}"