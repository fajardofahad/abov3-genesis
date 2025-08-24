"""
ABOV3 Genesis Agent Manager
Manages AI agents and their configurations
"""

import asyncio
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class Agent:
    """Agent configuration data class"""
    name: str
    model: str
    description: str
    system_prompt: str
    created: str = None
    modified: str = None
    usage_count: int = 0
    
    def __post_init__(self):
        if self.created is None:
            self.created = datetime.now().isoformat()
        if self.modified is None:
            self.modified = self.created

class AgentManager:
    """
    Agent Manager for ABOV3 Genesis
    Handles creation, management, and switching of AI agents
    """
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.abov3_dir = self.project_path / '.abov3'
        self.agents_dir = self.abov3_dir / 'agents'
        self.current_agent_file = self.agents_dir / 'current.yaml'
        
        # Ensure directories exist
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Agent storage
        self.agents: Dict[str, Agent] = {}
        self.current_agent: Optional[Agent] = None
        
        # Load existing agents
        asyncio.create_task(self._load_agents()) if self._is_async_context() else self._load_agents_sync()
    
    def _is_async_context(self) -> bool:
        """Check if we're in an async context"""
        try:
            import asyncio
            asyncio.current_task()
            return True
        except RuntimeError:
            return False
    
    def _load_agents_sync(self):
        """Load agents synchronously"""
        # Load existing agent files
        for agent_file in self.agents_dir.glob('*.yaml'):
            if agent_file.name != 'current.yaml':
                try:
                    with open(agent_file, 'r') as f:
                        agent_data = yaml.safe_load(f)
                    
                    if agent_data:
                        agent = Agent(**agent_data)
                        self.agents[agent.name] = agent
                except Exception as e:
                    print(f"Error loading agent from {agent_file}: {e}")
        
        # Load current agent
        if self.current_agent_file.exists():
            try:
                with open(self.current_agent_file, 'r') as f:
                    current_data = yaml.safe_load(f)
                
                current_name = current_data.get('current_agent')
                if current_name and current_name in self.agents:
                    self.current_agent = self.agents[current_name]
            except Exception as e:
                print(f"Error loading current agent: {e}")
        
        # Set default agent if none is current
        if not self.current_agent and self.agents:
            self.current_agent = list(self.agents.values())[0]
        elif not self.agents:
            # Create default Genesis agents
            self._create_default_agents()
    
    async def _load_agents(self):
        """Load agents asynchronously"""
        self._load_agents_sync()
    
    def _create_default_agents(self):
        """Create default Genesis agents"""
        default_agents = [
            {
                'name': 'genesis-architect',
                'model': 'llama3:latest',
                'description': 'Transforms ideas into system architecture',
                'system_prompt': """You are the Genesis Architect, a specialized AI agent focused on transforming ideas into comprehensive system architectures.

Your expertise includes:
- Breaking down complex ideas into manageable components
- Designing scalable and maintainable architectures
- Creating detailed technical specifications
- Planning implementation phases and milestones
- Selecting appropriate technology stacks
- Identifying potential challenges and solutions

When given an idea, you:
1. Analyze the core requirements and goals
2. Design the overall system architecture
3. Define key components and their relationships
4. Recommend technology choices with justification
5. Create implementation roadmaps
6. Identify risks and mitigation strategies

Focus on practical, buildable designs that follow best practices and industry standards."""
            },
            {
                'name': 'genesis-builder',
                'model': 'codellama:latest', 
                'description': 'Builds complete applications from scratch',
                'system_prompt': """You are the Genesis Builder, a specialized AI agent focused on transforming designs and specifications into working code.

Your expertise includes:
- Writing production-ready, clean code
- Implementing complete features and applications
- Following coding best practices and patterns
- Creating modular, maintainable code structures
- Handling error cases and edge conditions
- Writing comprehensive documentation

When building applications, you:
1. Follow the provided architecture and specifications exactly
2. Write complete, functional code (not just snippets)
3. Include proper error handling and validation
4. Add meaningful comments and documentation
5. Use consistent coding styles and patterns
6. Create necessary configuration files
7. Suggest testing strategies

Always prioritize code quality, security, and maintainability. Generate complete, ready-to-run code that users can implement immediately."""
            },
            {
                'name': 'genesis-designer',
                'model': 'llama3:latest',
                'description': 'Creates beautiful UIs from descriptions',
                'system_prompt': """You are the Genesis Designer, a specialized AI agent focused on creating beautiful, functional user interfaces.

Your expertise includes:
- Designing intuitive user experiences (UX)
- Creating attractive user interfaces (UI)  
- Implementing responsive, accessible designs
- Following modern design principles and trends
- Creating consistent design systems
- Optimizing for different devices and screen sizes

When designing interfaces, you:
1. Focus on user needs and workflows
2. Create clean, modern, and accessible designs
3. Use appropriate color schemes and typography
4. Implement responsive layouts
5. Follow platform-specific design guidelines
6. Create reusable component systems
7. Consider performance and loading times

Generate complete UI code with HTML, CSS, and JavaScript as needed. Always prioritize user experience, accessibility, and visual appeal."""
            },
            {
                'name': 'genesis-optimizer',
                'model': 'llama3:latest',
                'description': 'Optimizes and perfects existing code',
                'system_prompt': """You are the Genesis Optimizer, a specialized AI agent focused on improving and perfecting existing code and applications.

Your expertise includes:
- Code review and quality analysis
- Performance optimization and profiling
- Security vulnerability assessment
- Code refactoring and modernization
- Best practices implementation
- Technical debt reduction

When optimizing code, you:
1. Analyze existing code for improvements
2. Identify performance bottlenecks
3. Suggest security enhancements
4. Refactor for better maintainability
5. Recommend modern patterns and practices
6. Optimize for speed, memory, and resources
7. Improve code readability and documentation

Always provide specific, actionable improvements with clear explanations of the benefits."""
            },
            {
                'name': 'genesis-deployer',
                'model': 'llama3:latest',
                'description': 'Handles deployment and production readiness',
                'system_prompt': """You are the Genesis Deployer, a specialized AI agent focused on taking applications from development to production.

Your expertise includes:
- Deployment strategy planning
- Cloud platform configuration
- CI/CD pipeline setup
- Environment management
- Monitoring and logging setup
- Security and compliance considerations

When handling deployment, you:
1. Assess application readiness for production
2. Design deployment architectures
3. Configure cloud services and infrastructure
4. Set up automated deployment pipelines
5. Implement monitoring and alerting
6. Plan scaling and performance strategies
7. Ensure security and compliance requirements

Focus on reliable, scalable deployment solutions that follow industry best practices."""
            }
        ]
        
        for agent_config in default_agents:
            agent = Agent(**agent_config)
            self.agents[agent.name] = agent
            self._save_agent(agent)
        
        # Set the first agent as current
        if self.agents:
            self.current_agent = list(self.agents.values())[0]
            self._save_current_agent()
    
    def _save_agent(self, agent: Agent):
        """Save an agent to file"""
        agent_file = self.agents_dir / f'{agent.name}.yaml'
        try:
            with open(agent_file, 'w') as f:
                yaml.dump(asdict(agent), f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving agent {agent.name}: {e}")
    
    def _save_current_agent(self):
        """Save current agent reference"""
        current_data = {
            'current_agent': self.current_agent.name if self.current_agent else None,
            'updated': datetime.now().isoformat()
        }
        
        try:
            with open(self.current_agent_file, 'w') as f:
                yaml.dump(current_data, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving current agent: {e}")
    
    async def create_agent(
        self, 
        name: str, 
        model: str, 
        description: str, 
        system_prompt: str
    ) -> bool:
        """Create a new agent"""
        if name in self.agents:
            return False  # Agent already exists
        
        agent = Agent(
            name=name,
            model=model,
            description=description,
            system_prompt=system_prompt
        )
        
        self.agents[name] = agent
        self._save_agent(agent)
        
        # Set as current if it's the first agent
        if not self.current_agent:
            self.current_agent = agent
            self._save_current_agent()
        
        return True
    
    async def create_agent_from_config(self, config: Dict[str, Any]) -> bool:
        """Create agent from configuration dictionary"""
        required_fields = ['name', 'model', 'description', 'system_prompt']
        
        if not all(field in config for field in required_fields):
            return False
        
        return await self.create_agent(
            config['name'],
            config['model'],
            config['description'],
            config['system_prompt']
        )
    
    async def update_agent(
        self, 
        name: str, 
        model: str = None, 
        description: str = None, 
        system_prompt: str = None
    ) -> bool:
        """Update an existing agent"""
        if name not in self.agents:
            return False
        
        agent = self.agents[name]
        
        if model is not None:
            agent.model = model
        if description is not None:
            agent.description = description
        if system_prompt is not None:
            agent.system_prompt = system_prompt
        
        agent.modified = datetime.now().isoformat()
        
        self._save_agent(agent)
        return True
    
    async def delete_agent(self, name: str) -> bool:
        """Delete an agent"""
        if name not in self.agents:
            return False
        
        # Don't delete if it's the current agent and the only one
        if self.current_agent and self.current_agent.name == name and len(self.agents) == 1:
            return False
        
        # Remove from memory
        del self.agents[name]
        
        # Delete file
        agent_file = self.agents_dir / f'{name}.yaml'
        if agent_file.exists():
            agent_file.unlink()
        
        # Switch to another agent if this was current
        if self.current_agent and self.current_agent.name == name:
            if self.agents:
                self.current_agent = list(self.agents.values())[0]
                self._save_current_agent()
            else:
                self.current_agent = None
        
        return True
    
    async def switch_agent(self, name: str) -> bool:
        """Switch to a different agent"""
        if name not in self.agents:
            return False
        
        # Update usage count for previous agent
        if self.current_agent:
            self.current_agent.usage_count += 1
            self._save_agent(self.current_agent)
        
        self.current_agent = self.agents[name]
        self._save_current_agent()
        return True
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name"""
        return self.agents.get(name)
    
    def get_available_agents(self) -> List[Agent]:
        """Get all available agents"""
        return list(self.agents.values())
    
    def get_genesis_agents(self) -> List[Agent]:
        """Get Genesis-specific agents"""
        genesis_agent_names = [
            'genesis-architect', 'genesis-builder', 'genesis-designer',
            'genesis-optimizer', 'genesis-deployer'
        ]
        return [agent for agent in self.agents.values() if agent.name in genesis_agent_names]
    
    def has_agent(self, name: str) -> bool:
        """Check if an agent exists"""
        return name in self.agents
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about agents"""
        return {
            'total_agents': len(self.agents),
            'genesis_agents': len(self.get_genesis_agents()),
            'current_agent': self.current_agent.name if self.current_agent else None,
            'most_used': max(self.agents.values(), key=lambda a: a.usage_count).name if self.agents else None,
            'agents': [
                {
                    'name': agent.name,
                    'model': agent.model,
                    'description': agent.description,
                    'usage_count': agent.usage_count,
                    'created': agent.created
                }
                for agent in self.agents.values()
            ]
        }
    
    async def import_agent(self, agent_config: Dict[str, Any]) -> bool:
        """Import an agent from configuration"""
        return await self.create_agent_from_config(agent_config)
    
    def export_agent(self, name: str) -> Optional[Dict[str, Any]]:
        """Export an agent configuration"""
        agent = self.get_agent(name)
        if agent:
            return asdict(agent)
        return None
    
    def export_all_agents(self) -> Dict[str, Any]:
        """Export all agents"""
        return {
            'agents': [asdict(agent) for agent in self.agents.values()],
            'current_agent': self.current_agent.name if self.current_agent else None,
            'exported': datetime.now().isoformat()
        }
    
    async def reset_to_defaults(self):
        """Reset to default Genesis agents"""
        # Clear existing agents
        self.agents.clear()
        self.current_agent = None
        
        # Remove agent files
        for agent_file in self.agents_dir.glob('*.yaml'):
            agent_file.unlink()
        
        # Recreate defaults
        self._create_default_agents()
    
    def get_recommended_models(self) -> List[str]:
        """Get recommended models for agents"""
        return [
            'llama3:latest',
            'codellama:latest',
            'gemma:7b',
            'mistral:latest',
            'deepseek-coder:6.7b',
            'qwen:14b',
            'wizard-coder:latest'
        ]
    
    async def validate_agent_config(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """Validate agent configuration"""
        required_fields = ['name', 'model', 'description', 'system_prompt']
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"
        
        # Check field types
        if not isinstance(config['name'], str) or not config['name'].strip():
            return False, "Name must be a non-empty string"
        
        if not isinstance(config['model'], str) or not config['model'].strip():
            return False, "Model must be a non-empty string"
        
        if not isinstance(config['description'], str) or not config['description'].strip():
            return False, "Description must be a non-empty string"
        
        if not isinstance(config['system_prompt'], str) or not config['system_prompt'].strip():
            return False, "System prompt must be a non-empty string"
        
        # Check name format
        name = config['name'].strip()
        if not name.replace('-', '').replace('_', '').isalnum():
            return False, "Name must contain only letters, numbers, hyphens, and underscores"
        
        # Check if name already exists
        if name in self.agents:
            return False, f"Agent with name '{name}' already exists"
        
        return True, "Valid configuration"