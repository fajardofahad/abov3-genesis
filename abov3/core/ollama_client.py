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
    
    def get_genesis_optimized_options(self, task_type: str = "general", model_name: str = None) -> Dict[str, Any]:
        """Get optimized options for different Genesis tasks and models"""
        # Base options optimized for code generation quality
        base_options = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 50,
            "repeat_penalty": 1.02,
            "num_predict": -1,  # No limit on prediction length
            "stop": ["<|endoftext|>", "<|end|>", "Human:", "User:"],
            "seed": -1  # Random seed for variety
        }
        
        # Task-specific optimizations based on research and testing
        task_optimizations = {
            "code_generation": {
                "temperature": 0.1,  # Very low for deterministic code
                "top_p": 0.95,
                "top_k": 40,
                "repeat_penalty": 1.0,  # No penalty for code patterns
                "mirostat": 2,  # Enable mirostat for better coherence
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1
            },
            "code_review": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 30,
                "repeat_penalty": 1.1
            },
            "debugging": {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "repeat_penalty": 1.0
            },
            "architecture_design": {
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 50,
                "repeat_penalty": 1.1
            },
            "explanation": {
                "temperature": 0.4,
                "top_p": 0.9,
                "top_k": 50,
                "repeat_penalty": 1.2
            },
            "creative_writing": {
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 80,
                "repeat_penalty": 1.3
            },
            "analysis": {
                "temperature": 0.2,
                "top_p": 0.85,
                "top_k": 30,
                "repeat_penalty": 1.1
            },
            "conversation": {
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": 60,
                "repeat_penalty": 1.1
            }
        }
        
        # Model-specific optimizations
        model_optimizations = {
            "codellama": {
                "temperature": 0.05,  # Extremely low for best code quality
                "top_k": 20,
                "repeat_penalty": 0.95  # Allow code repetition
            },
            "deepseek-coder": {
                "temperature": 0.1,
                "top_p": 0.98,
                "repeat_penalty": 1.0
            },
            "starcoder": {
                "temperature": 0.1,
                "top_k": 30,
                "repeat_penalty": 1.0
            },
            "llama3": {
                "temperature": 0.2,
                "mirostat": 1,
                "mirostat_tau": 3.0
            },
            "qwen": {
                "temperature": 0.15,
                "top_p": 0.95,
                "top_k": 35
            }
        }
        
        # Apply task optimizations
        if task_type in task_optimizations:
            base_options.update(task_optimizations[task_type])
        
        # Apply model-specific optimizations
        if model_name:
            for model_key, opts in model_optimizations.items():
                if model_key.lower() in model_name.lower():
                    base_options.update(opts)
                    break
        
        return base_options