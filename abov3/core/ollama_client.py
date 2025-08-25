"""
ABOV3 Genesis - Ollama Client
Interface for communicating with local Ollama models
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncGenerator
import aiohttp
from pathlib import Path

class OllamaClient:
    """
    Ollama API client for ABOV3 Genesis
    Handles communication with local Ollama server
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.available_models = []
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def connect(self):
        """Connect to Ollama server"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Test connection and get available models
        await self.refresh_models()
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/version", timeout=5) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def refresh_models(self):
        """Refresh the list of available models"""
        try:
            if not self.session:
                await self.connect()
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    self.available_models = data.get('models', [])
                else:
                    self.available_models = []
        except Exception as e:
            print(f"Error refreshing models: {e}")
            self.available_models = []
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        await self.refresh_models()
        return self.available_models
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        try:
            if not self.session:
                await self.connect()
            
            async with self.session.post(
                f"{self.base_url}/api/show", 
                json={"name": model_name}
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            print(f"Error getting model info: {e}")
        
        return None
    
    async def generate(
        self, 
        model: str, 
        prompt: str, 
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate text using Ollama model"""
        if not self.session:
            await self.connect()
        
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if system:
            payload["system"] = system
        
        if context:
            payload["context"] = context
        
        if options:
            payload["options"] = options
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    if stream:
                        # Streaming response
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    yield data
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # Non-streaming response
                        data = await response.json()
                        yield data
                else:
                    error_text = await response.text()
                    yield {
                        "error": f"HTTP {response.status}: {error_text}",
                        "done": True
                    }
        except Exception as e:
            yield {
                "error": f"Request failed: {str(e)}",
                "done": True
            }
    
    async def chat(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Chat with Ollama model using messages format"""
        if not self.session:
            await self.connect()
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        if options:
            payload["options"] = options
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                if response.status == 200:
                    if stream:
                        # Streaming response
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    yield data
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # Non-streaming response
                        data = await response.json()
                        yield data
                else:
                    error_text = await response.text()
                    yield {
                        "error": f"HTTP {response.status}: {error_text}",
                        "done": True
                    }
        except Exception as e:
            yield {
                "error": f"Request failed: {str(e)}",
                "done": True
            }
    
    async def pull_model(self, model_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Pull/download a model"""
        if not self.session:
            await self.connect()
        
        payload = {"name": model_name}
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json=payload
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                yield data
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.text()
                    yield {
                        "error": f"HTTP {response.status}: {error_text}",
                        "done": True
                    }
        except Exception as e:
            yield {
                "error": f"Request failed: {str(e)}",
                "done": True
            }
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model"""
        if not self.session:
            await self.connect()
        
        try:
            async with self.session.delete(
                f"{self.base_url}/api/delete",
                json={"name": model_name}
            ) as response:
                return response.status == 200
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False
    
    async def copy_model(self, source: str, destination: str) -> bool:
        """Copy a model"""
        if not self.session:
            await self.connect()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/copy",
                json={"source": source, "destination": destination}
            ) as response:
                return response.status == 200
        except Exception as e:
            print(f"Error copying model: {e}")
            return False
    
    def get_recommended_models(self) -> List[Dict[str, Any]]:
        """Get recommended models for Genesis"""
        return [
            {
                "name": "llama3:latest",
                "size": "4.7GB",
                "description": "Meta's Llama 3 - Excellent for general tasks",
                "use_case": "General purpose, reasoning, writing"
            },
            {
                "name": "codellama:latest",
                "size": "3.8GB", 
                "description": "Code generation and understanding specialist",
                "use_case": "Code generation, debugging, refactoring"
            },
            {
                "name": "gemma:7b",
                "size": "5.2GB",
                "description": "Google's Gemma - Fast and efficient",
                "use_case": "Quick responses, general coding"
            },
            {
                "name": "mistral:latest",
                "size": "4.1GB",
                "description": "Mistral AI's model - Good balance",
                "use_case": "Balanced performance, multilingual"
            },
            {
                "name": "deepseek-coder:6.7b",
                "size": "3.8GB",
                "description": "Specialized coding model",
                "use_case": "Advanced code generation, architecture"
            }
        ]
    
    async def check_model_exists(self, model_name: str) -> bool:
        """Check if a model exists locally"""
        models = await self.list_models()
        return any(model.get('name') == model_name for model in models)
    
    async def get_model_size(self, model_name: str) -> Optional[int]:
        """Get the size of a model in bytes"""
        info = await self.get_model_info(model_name)
        return info.get('size') if info else None
    
    async def embed(self, model: str, prompt: str) -> Optional[List[float]]:
        """Generate embeddings for text"""
        if not self.session:
            await self.connect()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": prompt}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('embedding')
        except Exception as e:
            print(f"Error generating embeddings: {e}")
        
        return None
    
    def get_genesis_optimized_options(self, task_type: str = "general", model_name: str = None, context_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get optimized options for different Genesis tasks and models with advanced Claude-level optimization"""
        context_info = context_info or {}
        
        # Base options optimized for Claude-level performance
        base_options = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 50,
            "repeat_penalty": 1.02,
            "num_predict": -1,  # No limit on prediction length
            "stop": ["<|endoftext|>", "<|end|>", "Human:", "User:", "\n\nHuman:", "\n\nUser:"],
            "seed": -1,  # Random seed for variety
            "mirostat": 0,  # Disable by default, enable for specific cases
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1
        }
        
        # Advanced task-specific optimizations for Claude-level performance
        task_optimizations = {
            "code_generation": {
                "temperature": 0.05,  # Extremely low for consistent high-quality code
                "top_p": 0.98,
                "top_k": 35,
                "repeat_penalty": 0.98,  # Allow code patterns and consistency
                "mirostat": 2,  # Enable mirostat for better coherence
                "mirostat_tau": 4.0,
                "mirostat_eta": 0.05,  # Lower eta for more stable generation
                "stop": ["\n\n\n\n", "```\n\n", "<|end|>", "Human:"],
                "context_length": 16384  # Prefer longer context for code
            },
            "code_review": {
                "temperature": 0.15,  # Low for consistent analysis
                "top_p": 0.92,
                "top_k": 25,
                "repeat_penalty": 1.05,
                "mirostat": 1,  # Better for analytical tasks
                "mirostat_tau": 3.0,
                "stop": ["\n\n---", "<|end|>", "Human:"]
            },
            "debugging": {
                "temperature": 0.08,  # Very low for precise debugging
                "top_p": 0.96,
                "top_k": 30,
                "repeat_penalty": 1.0,
                "mirostat": 2,
                "mirostat_tau": 4.5,
                "stop": ["\n\n\n", "<|end|>", "Human:", "Next issue:"]
            },
            "architecture": {
                "temperature": 0.25,  # Moderate for creative but structured thinking
                "top_p": 0.88,
                "top_k": 45,
                "repeat_penalty": 1.08,
                "mirostat": 1,
                "mirostat_tau": 6.0,  # Higher tau for more diverse architecture options
                "stop": ["\n\n---", "<|end|>", "Human:"]
            },
            "explanation": {
                "temperature": 0.3,  # Moderate for clear, varied explanations
                "top_p": 0.9,
                "top_k": 50,
                "repeat_penalty": 1.15,
                "mirostat": 0,  # Disable mirostat for explanations
                "stop": ["\n\n\n", "<|end|>", "Human:", "Next question:"]
            },
            "code_completion": {
                "temperature": 0.02,  # Extremely low for predictable completions
                "top_p": 0.99,
                "top_k": 20,
                "repeat_penalty": 0.95,  # Allow repetitive patterns in code
                "mirostat": 2,
                "mirostat_tau": 2.0,
                "stop": ["\n\n", "```", "<|end|>"],
                "num_predict": 200  # Limit for completions
            },
            "optimization": {
                "temperature": 0.12,
                "top_p": 0.94,
                "top_k": 35,
                "repeat_penalty": 1.05,
                "mirostat": 2,
                "mirostat_tau": 4.0,
                "stop": ["\n\n---", "<|end|>", "Human:"]
            },
            "testing": {
                "temperature": 0.1,
                "top_p": 0.94,
                "top_k": 35,
                "repeat_penalty": 1.03,
                "mirostat": 1,
                "mirostat_tau": 3.5,
                "stop": ["\n\n---", "<|end|>", "Human:"]
            },
            "creative_writing": {
                "temperature": 0.75,  # High for creativity but not too chaotic
                "top_p": 0.92,
                "top_k": 70,
                "repeat_penalty": 1.25
            },
            "analysis": {
                "temperature": 0.18,  # Low for consistent analysis
                "top_p": 0.88,
                "top_k": 30,
                "repeat_penalty": 1.08,
                "mirostat": 1,
                "mirostat_tau": 3.5
            },
            "conversation": {
                "temperature": 0.4,  # Moderate for natural but consistent conversation
                "top_p": 0.92,
                "top_k": 55,
                "repeat_penalty": 1.12,
                "stop": ["\n\nHuman:", "\n\nUser:", "<|end|>"]
            },
            "refactoring": {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 35,
                "repeat_penalty": 1.02,
                "mirostat": 2,
                "mirostat_tau": 4.0
            },
            "api_design": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.08,
                "mirostat": 1,
                "mirostat_tau": 5.0
            },
            "documentation_generation": {
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 45,
                "repeat_penalty": 1.15,
                "stop": ["\n\n---", "<|end|>", "# ", "## "]
            }
        }
        
        # Advanced model-specific optimizations for Claude-level performance
        model_optimizations = {
            "codellama": {
                "temperature": 0.03,  # Extremely low for best code quality
                "top_p": 0.98,
                "top_k": 15,
                "repeat_penalty": 0.92,  # Allow code repetition and patterns
                "mirostat": 2,
                "mirostat_tau": 3.0,
                "mirostat_eta": 0.05
            },
            "deepseek-coder": {
                "temperature": 0.08,
                "top_p": 0.99,
                "top_k": 25,
                "repeat_penalty": 0.98,
                "mirostat": 2,
                "mirostat_tau": 4.0,
                "mirostat_eta": 0.05
            },
            "starcoder": {
                "temperature": 0.06,
                "top_p": 0.97,
                "top_k": 25,
                "repeat_penalty": 0.95,
                "mirostat": 2,
                "mirostat_tau": 3.5
            },
            "llama3": {
                "temperature": 0.15,
                "top_p": 0.92,
                "top_k": 40,
                "mirostat": 1,
                "mirostat_tau": 4.0,
                "mirostat_eta": 0.1,
                "repeat_penalty": 1.05
            },
            "qwen": {
                "temperature": 0.12,
                "top_p": 0.96,
                "top_k": 32,
                "repeat_penalty": 1.03,
                "mirostat": 1,
                "mirostat_tau": 3.5
            },
            "mistral": {
                "temperature": 0.18,
                "top_p": 0.9,
                "top_k": 45,
                "repeat_penalty": 1.08,
                "mirostat": 1,
                "mirostat_tau": 4.0
            },
            "gemma": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 50,
                "repeat_penalty": 1.1
            },
            "phi": {
                "temperature": 0.15,
                "top_p": 0.93,
                "top_k": 35,
                "repeat_penalty": 1.05
            }
        }
        
        # Apply task optimizations first
        if task_type in task_optimizations:
            base_options.update(task_optimizations[task_type])
        
        # Apply model-specific optimizations
        if model_name:
            for model_key, opts in model_optimizations.items():
                if model_key.lower() in model_name.lower():
                    base_options.update(opts)
                    break
        
        # Apply context-aware optimizations for Claude-level performance
        if context_info:
            # Adjust for complexity level
            complexity = context_info.get('complexity_level', 'medium')
            if complexity == 'high':
                base_options['temperature'] = max(0.02, base_options['temperature'] * 0.8)
                base_options['mirostat_tau'] = base_options.get('mirostat_tau', 4.0) + 1.0
                base_options['top_p'] = min(0.99, base_options.get('top_p', 0.95) + 0.02)
            elif complexity == 'low':
                base_options['temperature'] = min(0.3, base_options['temperature'] * 1.2)
            
            # Adjust for performance requirements
            if context_info.get('performance_critical', False):
                base_options['temperature'] = max(0.01, base_options['temperature'] * 0.5)
                base_options['top_p'] = min(0.99, base_options.get('top_p', 0.95) + 0.05)
                base_options['mirostat'] = 2  # Force mirostat for consistency
            
            # Adjust for creativity requirements
            if context_info.get('creative_task', False):
                base_options['temperature'] = min(0.6, base_options['temperature'] * 2.0)
                base_options['top_k'] = min(100, base_options.get('top_k', 50) + 20)
            
            # Adjust for code completion tasks
            if context_info.get('is_completion', False):
                base_options['temperature'] = max(0.01, base_options['temperature'] * 0.3)
                base_options['num_predict'] = min(500, base_options.get('num_predict', -1))
        
        return base_options