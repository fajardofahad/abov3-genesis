"""
ABOV3 Genesis - Secure AI Manager
Advanced security layer for AI model interactions
"""

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class AISecurityLevel(Enum):
    """AI security levels"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


class SecureAIManager:
    """
    Secure AI interaction manager
    Provides comprehensive protection for AI model interactions
    """
    
    def __init__(self, max_prompt_length: int = 10000, max_response_length: int = 50000,
                 enable_content_filtering: bool = True, prompt_guard=None, audit_logger=None):
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.enable_content_filtering = enable_content_filtering
        self.prompt_guard = prompt_guard
        self.audit_logger = audit_logger
        
        # Security configuration
        self.security_level = AISecurityLevel.ENHANCED
        
        # Response filtering patterns
        self.dangerous_response_patterns = [
            r'(?i)here\'s how to hack',
            r'(?i)steps to create malware',
            r'(?i)how to exploit',
            r'(?i)illegal activities',
            r'(?i)harmful instructions'
        ]
        
        # Statistics
        self.ai_stats = {
            'total_interactions': 0,
            'blocked_prompts': 0,
            'filtered_responses': 0,
            'security_violations': 0
        }
    
    async def secure_interaction(self, prompt: str, model: str, 
                               options: Optional[Dict[str, Any]] = None,
                               user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Secure AI model interaction"""
        self.ai_stats['total_interactions'] += 1
        
        try:
            # Validate prompt
            prompt_result = await self._validate_prompt(prompt, user_context)
            if not prompt_result['valid']:
                self.ai_stats['blocked_prompts'] += 1
                return {
                    'success': False,
                    'error': 'Prompt validation failed',
                    'reason': prompt_result['reason'],
                    'security_flags': prompt_result.get('flags', [])
                }
            
            # Sanitize prompt if needed
            sanitized_prompt = prompt_result.get('sanitized_prompt', prompt)
            
            # Validate model selection
            model_result = await self._validate_model(model, user_context)
            if not model_result['valid']:
                return {
                    'success': False,
                    'error': 'Model validation failed',
                    'reason': model_result['reason']
                }
            
            # Secure model options
            secure_options = await self._secure_model_options(options or {})
            
            # This would integrate with the actual AI model
            # For now, simulate a response
            ai_response = await self._simulate_ai_response(sanitized_prompt, model, secure_options)
            
            # Validate and filter response
            response_result = await self._validate_response(ai_response, user_context)
            if not response_result['valid']:
                self.ai_stats['filtered_responses'] += 1
                return {
                    'success': False,
                    'error': 'Response validation failed',
                    'reason': response_result['reason']
                }
            
            # Log successful interaction
            if self.audit_logger:
                await self.audit_logger.log_event("secure_ai_interaction", {
                    "model": model,
                    "prompt_length": len(sanitized_prompt),
                    "response_length": len(response_result['response']),
                    "security_level": self.security_level.value,
                    "user_id": user_context.get('user_id') if user_context else None
                })
            
            return {
                'success': True,
                'response': response_result['response'],
                'model': model,
                'security_applied': prompt_result.get('security_applied', []),
                'metadata': {
                    'prompt_validated': True,
                    'response_filtered': response_result.get('filtered', False),
                    'security_level': self.security_level.value
                }
            }
            
        except Exception as e:
            self.ai_stats['security_violations'] += 1
            return {
                'success': False,
                'error': f'Secure AI interaction failed: {str(e)}'
            }
    
    async def _validate_prompt(self, prompt: str, user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate AI prompt for security issues"""
        result = {
            'valid': True,
            'reason': '',
            'sanitized_prompt': prompt,
            'security_applied': [],
            'flags': []
        }
        
        # Length validation
        if len(prompt) > self.max_prompt_length:
            result['valid'] = False
            result['reason'] = f'Prompt exceeds maximum length of {self.max_prompt_length}'
            return result
        
        # Use prompt guard if available
        if self.prompt_guard:
            injection_result = await self.prompt_guard.analyze_prompt(prompt, user_context)
            if injection_result['is_injection']:
                result['valid'] = False
                result['reason'] = 'Prompt injection detected'
                result['flags'] = injection_result['injection_types']
                return result
            
            if injection_result['risk_score'] > 50:
                result['sanitized_prompt'] = injection_result['sanitized_prompt']
                result['security_applied'].append('prompt_sanitization')
        
        # Content filtering
        if self.enable_content_filtering:
            filtered_prompt = await self._filter_prompt_content(prompt)
            if filtered_prompt != prompt:
                result['sanitized_prompt'] = filtered_prompt
                result['security_applied'].append('content_filtering')
        
        return result
    
    async def _validate_model(self, model: str, user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate model selection and permissions"""
        result = {'valid': True, 'reason': ''}
        
        # Model whitelist
        allowed_models = [
            'llama3:latest', 'codellama:latest', 'mistral:latest',
            'gemma:7b', 'deepseek-coder:6.7b'
        ]
        
        if model not in allowed_models:
            result['valid'] = False
            result['reason'] = f'Model {model} not in allowed list'
            return result
        
        # User permission check (if user context provided)
        if user_context and user_context.get('user_role') == 'guest':
            basic_models = ['llama3:latest', 'gemma:7b']
            if model not in basic_models:
                result['valid'] = False
                result['reason'] = 'Insufficient permissions for advanced model'
        
        return result
    
    async def _secure_model_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Secure and validate model options"""
        secure_options = options.copy()
        
        # Enforce safe temperature ranges
        if 'temperature' in secure_options:
            temp = secure_options['temperature']
            if temp < 0 or temp > 1.5:
                secure_options['temperature'] = max(0, min(1.5, temp))
        
        # Limit response length
        if 'max_tokens' in secure_options:
            secure_options['max_tokens'] = min(secure_options['max_tokens'], 4096)
        
        # Remove potentially dangerous options
        dangerous_options = ['system_override', 'debug_mode', 'unsafe_mode']
        for option in dangerous_options:
            secure_options.pop(option, None)
        
        return secure_options
    
    async def _simulate_ai_response(self, prompt: str, model: str, options: Dict[str, Any]) -> str:
        """Simulate AI response (in real implementation, this would call the actual model)"""
        # This is a placeholder - in real implementation would call Ollama
        return f"This is a simulated secure response to: {prompt[:50]}..."
    
    async def _validate_response(self, response: str, user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate AI response for security issues"""
        result = {
            'valid': True,
            'reason': '',
            'response': response,
            'filtered': False
        }
        
        # Length validation
        if len(response) > self.max_response_length:
            result['valid'] = False
            result['reason'] = f'Response exceeds maximum length of {self.max_response_length}'
            return result
        
        # Content filtering
        if self.enable_content_filtering:
            import re
            for pattern in self.dangerous_response_patterns:
                if re.search(pattern, response):
                    result['valid'] = False
                    result['reason'] = 'Response contains potentially harmful content'
                    return result
        
        # Filter sensitive information
        filtered_response = await self._filter_sensitive_info(response)
        if filtered_response != response:
            result['response'] = filtered_response
            result['filtered'] = True
        
        return result
    
    async def _filter_prompt_content(self, prompt: str) -> str:
        """Filter dangerous content from prompt"""
        # Basic content filtering
        dangerous_keywords = [
            'hack', 'exploit', 'malware', 'virus', 'illegal'
        ]
        
        filtered_prompt = prompt
        for keyword in dangerous_keywords:
            filtered_prompt = filtered_prompt.replace(keyword, '[FILTERED]')
        
        return filtered_prompt
    
    async def _filter_sensitive_info(self, response: str) -> str:
        """Filter sensitive information from response"""
        import re
        
        # Filter potential API keys
        response = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[API_KEY_FILTERED]', response)
        
        # Filter potential passwords
        response = re.sub(r'password:\s*\S+', 'password: [FILTERED]', response, flags=re.IGNORECASE)
        
        # Filter email addresses in certain contexts
        response = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_FILTERED]', response)
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get AI security statistics"""
        return self.ai_stats.copy()
    
    def set_security_level(self, level: AISecurityLevel):
        """Set AI security level"""
        self.security_level = level
        
        # Adjust parameters based on security level
        if level == AISecurityLevel.MAXIMUM:
            self.max_prompt_length = 5000
            self.max_response_length = 25000
            self.enable_content_filtering = True
        elif level == AISecurityLevel.BASIC:
            self.max_prompt_length = 20000
            self.max_response_length = 100000
            self.enable_content_filtering = False