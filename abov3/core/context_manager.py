"""
ABOV3 Genesis - Smart Context Window Management System
Intelligent context management for optimal Ollama model performance
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import logging
from collections import deque, OrderedDict
import hashlib
from enum import Enum

logger = logging.getLogger(__name__)

class ContextPriority(Enum):
    """Context priority levels"""
    CRITICAL = 10       # Always included (system prompts, current request)
    HIGH = 8           # High-value context (recent code, project info)
    MEDIUM = 6         # Useful context (examples, docs)
    LOW = 4            # Nice-to-have (historical context)
    MINIMAL = 2        # Filler content

class ContentType(Enum):
    """Types of content in context"""
    SYSTEM_PROMPT = "system_prompt"
    USER_REQUEST = "user_request"
    CODE_EXAMPLE = "code_example"
    DOCUMENTATION = "documentation"
    PROJECT_INFO = "project_info"
    CONVERSATION_HISTORY = "conversation_history"
    ERROR_CONTEXT = "error_context"
    BEST_PRACTICES = "best_practices"
    ARCHITECTURE_GUIDE = "architecture_guide"

@dataclass
class ContextItem:
    """Individual context item with metadata"""
    content: str
    content_type: ContentType
    priority: ContextPriority
    tokens: int = 0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    relevance_score: float = 1.0
    usage_count: int = 0
    last_used: float = field(default_factory=lambda: datetime.now().timestamp())
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = self._estimate_tokens()
    
    def _estimate_tokens(self) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
        return max(1, len(self.content) // 4)
    
    def update_relevance(self, query: str, boost: float = 0.1):
        """Update relevance score based on query matching"""
        query_words = set(query.lower().split())
        content_words = set(self.content.lower().split())
        overlap = len(query_words.intersection(content_words))
        
        if overlap > 0:
            self.relevance_score += boost * overlap
            self.relevance_score = min(2.0, self.relevance_score)  # Cap at 2.0
    
    def mark_used(self):
        """Mark as recently used"""
        self.usage_count += 1
        self.last_used = datetime.now().timestamp()

class SmartContextManager:
    """
    Smart context window manager that optimizes content selection for different models
    """
    
    def __init__(self, max_context_tokens: int = 8192, model_name: str = None):
        self.max_context_tokens = max_context_tokens
        self.model_name = model_name
        
        # Context storage
        self.context_items: OrderedDict[str, ContextItem] = OrderedDict()
        self.context_cache: Dict[str, str] = {}
        
        # Usage statistics
        self.total_requests = 0
        self.cache_hits = 0
        
        # Model-specific optimizations
        self.model_context_preferences = {
            "codellama": {
                "prefer_code": True,
                "prefer_recent": True,
                "max_history": 3,
                "code_example_weight": 2.0
            },
            "deepseek-coder": {
                "prefer_code": True,
                "prefer_detailed": True,
                "max_history": 5,
                "documentation_weight": 1.5
            },
            "llama3": {
                "prefer_balanced": True,
                "prefer_conversation": True,
                "max_history": 8,
                "conversation_weight": 1.5
            },
            "qwen": {
                "prefer_structured": True,
                "prefer_examples": True,
                "max_history": 6,
                "example_weight": 1.8
            }
        }
        
        # Content type weights for different tasks
        self.task_content_weights = {
            "code_generation": {
                ContentType.CODE_EXAMPLE: 2.0,
                ContentType.BEST_PRACTICES: 1.5,
                ContentType.PROJECT_INFO: 1.3,
                ContentType.DOCUMENTATION: 1.2,
                ContentType.CONVERSATION_HISTORY: 0.8
            },
            "debugging": {
                ContentType.ERROR_CONTEXT: 2.5,
                ContentType.CODE_EXAMPLE: 2.0,
                ContentType.PROJECT_INFO: 1.5,
                ContentType.CONVERSATION_HISTORY: 1.3,
                ContentType.DOCUMENTATION: 1.0
            },
            "explanation": {
                ContentType.DOCUMENTATION: 2.0,
                ContentType.CODE_EXAMPLE: 1.8,
                ContentType.BEST_PRACTICES: 1.5,
                ContentType.CONVERSATION_HISTORY: 1.2,
                ContentType.PROJECT_INFO: 1.0
            },
            "architecture": {
                ContentType.ARCHITECTURE_GUIDE: 2.5,
                ContentType.PROJECT_INFO: 2.0,
                ContentType.BEST_PRACTICES: 1.8,
                ContentType.CODE_EXAMPLE: 1.5,
                ContentType.DOCUMENTATION: 1.3
            }
        }
    
    def add_context(
        self,
        content: str,
        content_type: ContentType,
        priority: ContextPriority = ContextPriority.MEDIUM,
        source: str = "unknown",
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add context item and return its ID"""
        # Generate unique ID
        context_id = hashlib.md5(f"{content[:100]}{datetime.now().timestamp()}".encode()).hexdigest()[:12]
        
        # Create context item
        item = ContextItem(
            content=content,
            content_type=content_type,
            priority=priority,
            source=source,
            metadata=metadata or {}
        )
        
        # Store context item
        self.context_items[context_id] = item
        
        # Maintain reasonable size
        if len(self.context_items) > 1000:
            self._cleanup_old_context()
        
        return context_id
    
    def add_system_prompt(self, prompt: str) -> str:
        """Add system prompt (always highest priority)"""
        return self.add_context(
            prompt,
            ContentType.SYSTEM_PROMPT,
            ContextPriority.CRITICAL,
            "system"
        )
    
    def add_user_request(self, request: str) -> str:
        """Add user request (always critical priority)"""
        return self.add_context(
            request,
            ContentType.USER_REQUEST,
            ContextPriority.CRITICAL,
            "user"
        )
    
    def add_code_example(self, code: str, description: str = "", language: str = "python") -> str:
        """Add code example with metadata"""
        formatted_code = f"```{language}\n{code}\n```"
        if description:
            formatted_code = f"{description}\n\n{formatted_code}"
        
        return self.add_context(
            formatted_code,
            ContentType.CODE_EXAMPLE,
            ContextPriority.HIGH,
            "code_example",
            {"language": language, "description": description}
        )
    
    def add_project_info(self, info: str) -> str:
        """Add project information"""
        return self.add_context(
            info,
            ContentType.PROJECT_INFO,
            ContextPriority.HIGH,
            "project"
        )
    
    def add_conversation_turn(self, role: str, message: str, timestamp: float = None) -> str:
        """Add conversation history item"""
        formatted_msg = f"{role}: {message}"
        return self.add_context(
            formatted_msg,
            ContentType.CONVERSATION_HISTORY,
            ContextPriority.MEDIUM,
            "conversation",
            {"role": role, "timestamp": timestamp or datetime.now().timestamp()}
        )
    
    def add_error_context(self, error_info: str) -> str:
        """Add error context for debugging"""
        return self.add_context(
            error_info,
            ContentType.ERROR_CONTEXT,
            ContextPriority.HIGH,
            "error"
        )
    
    def add_best_practices(self, practices: str, domain: str = "general") -> str:
        """Add best practices information"""
        return self.add_context(
            practices,
            ContentType.BEST_PRACTICES,
            ContextPriority.MEDIUM,
            "best_practices",
            {"domain": domain}
        )
    
    def build_optimized_context(
        self,
        task_type: str = "general",
        query: str = "",
        target_tokens: Optional[int] = None,
        include_history_items: int = 5
    ) -> str:
        """Build optimized context window for the request"""
        
        target_tokens = target_tokens or int(self.max_context_tokens * 0.8)  # Leave room for response
        
        # Cache key for this request
        cache_key = self._create_cache_key(task_type, query, target_tokens, include_history_items)
        if cache_key in self.context_cache:
            self.cache_hits += 1
            return self.context_cache[cache_key]
        
        self.total_requests += 1
        
        # Update relevance scores based on query
        if query:
            for item in self.context_items.values():
                item.update_relevance(query)
        
        # Get content weights for this task
        content_weights = self.task_content_weights.get(task_type, {})
        
        # Get model preferences
        model_prefs = self._get_model_preferences()
        
        # Score all items
        scored_items = []
        for item_id, item in self.context_items.items():
            score = self._calculate_item_score(item, content_weights, model_prefs, query)
            scored_items.append((score, item))
        
        # Sort by score (highest first)
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # Build context window
        selected_items = []
        current_tokens = 0
        
        # Always include critical priority items first
        for score, item in scored_items:
            if item.priority == ContextPriority.CRITICAL:
                selected_items.append(item)
                current_tokens += item.tokens
                item.mark_used()
        
        # Add high priority items
        for score, item in scored_items:
            if item.priority == ContextPriority.HIGH and current_tokens + item.tokens <= target_tokens:
                selected_items.append(item)
                current_tokens += item.tokens
                item.mark_used()
        
        # Add medium priority items if space available
        for score, item in scored_items:
            if item.priority == ContextPriority.MEDIUM and current_tokens + item.tokens <= target_tokens:
                # Limit conversation history based on include_history_items
                if item.content_type == ContentType.CONVERSATION_HISTORY:
                    history_count = sum(1 for i in selected_items if i.content_type == ContentType.CONVERSATION_HISTORY)
                    if history_count >= include_history_items:
                        continue
                
                selected_items.append(item)
                current_tokens += item.tokens
                item.mark_used()
        
        # Add low priority items if still space
        for score, item in scored_items:
            if item.priority == ContextPriority.LOW and current_tokens + item.tokens <= target_tokens:
                selected_items.append(item)
                current_tokens += item.tokens
                item.mark_used()
        
        # Order items logically for the model
        ordered_items = self._order_items_for_model(selected_items)
        
        # Build final context
        context_parts = []
        for item in ordered_items:
            if item.content_type == ContentType.SYSTEM_PROMPT:
                continue  # System prompts handled separately
            context_parts.append(item.content)
        
        final_context = "\n\n".join(context_parts)
        
        # Cache the result
        self.context_cache[cache_key] = final_context
        if len(self.context_cache) > 100:  # Limit cache size
            oldest_key = next(iter(self.context_cache))
            del self.context_cache[oldest_key]
        
        logger.debug(f"Built context: {current_tokens} tokens from {len(selected_items)} items")
        return final_context
    
    def _calculate_item_score(
        self,
        item: ContextItem,
        content_weights: Dict[ContentType, float],
        model_prefs: Dict[str, Any],
        query: str
    ) -> float:
        """Calculate score for context item"""
        
        base_score = item.priority.value  # Base score from priority
        
        # Content type weight
        content_weight = content_weights.get(item.content_type, 1.0)
        base_score *= content_weight
        
        # Model preference weight
        if model_prefs.get("prefer_code", False) and item.content_type == ContentType.CODE_EXAMPLE:
            base_score *= model_prefs.get("code_example_weight", 1.5)
        elif model_prefs.get("prefer_conversation", False) and item.content_type == ContentType.CONVERSATION_HISTORY:
            base_score *= model_prefs.get("conversation_weight", 1.5)
        elif model_prefs.get("prefer_detailed", False) and item.content_type == ContentType.DOCUMENTATION:
            base_score *= model_prefs.get("documentation_weight", 1.5)
        elif model_prefs.get("prefer_examples", False) and item.content_type == ContentType.CODE_EXAMPLE:
            base_score *= model_prefs.get("example_weight", 1.5)
        
        # Relevance score
        base_score *= item.relevance_score
        
        # Recency bonus (more recent = higher score)
        age_hours = (datetime.now().timestamp() - item.timestamp) / 3600
        recency_factor = max(0.1, 1.0 - (age_hours / 168))  # Decay over a week
        if model_prefs.get("prefer_recent", False):
            base_score *= (recency_factor * 2.0)  # Double recency weight
        else:
            base_score *= recency_factor
        
        # Usage frequency bonus
        usage_bonus = min(1.5, 1.0 + (item.usage_count * 0.1))
        base_score *= usage_bonus
        
        # Token efficiency (prefer more information per token)
        if item.tokens > 0:
            # Favor items with good information density
            content_density = len(item.content.split()) / item.tokens
            density_bonus = min(1.3, 1.0 + (content_density * 0.1))
            base_score *= density_bonus
        
        return base_score
    
    def _get_model_preferences(self) -> Dict[str, Any]:
        """Get model-specific preferences"""
        if not self.model_name:
            return {}
        
        for model_key, prefs in self.model_context_preferences.items():
            if model_key.lower() in self.model_name.lower():
                return prefs
        
        return {}
    
    def _order_items_for_model(self, items: List[ContextItem]) -> List[ContextItem]:
        """Order items in optimal sequence for model understanding"""
        ordered = []
        
        # Group by content type
        grouped = {}
        for item in items:
            if item.content_type not in grouped:
                grouped[item.content_type] = []
            grouped[item.content_type].append(item)
        
        # Optimal order for most models:
        # 1. Project info (sets context)
        # 2. Best practices (establishes standards)
        # 3. Architecture guides (high-level understanding)
        # 4. Documentation (detailed reference)
        # 5. Code examples (concrete implementations)
        # 6. Error context (specific problem context)
        # 7. Conversation history (recent context)
        # 8. User request (what to do now)
        
        order_preference = [
            ContentType.PROJECT_INFO,
            ContentType.BEST_PRACTICES,
            ContentType.ARCHITECTURE_GUIDE,
            ContentType.DOCUMENTATION,
            ContentType.CODE_EXAMPLE,
            ContentType.ERROR_CONTEXT,
            ContentType.CONVERSATION_HISTORY,
            ContentType.USER_REQUEST
        ]
        
        for content_type in order_preference:
            if content_type in grouped:
                # Sort items of same type by score (timestamp for history)
                if content_type == ContentType.CONVERSATION_HISTORY:
                    grouped[content_type].sort(key=lambda x: x.timestamp)
                else:
                    grouped[content_type].sort(key=lambda x: x.relevance_score, reverse=True)
                
                ordered.extend(grouped[content_type])
        
        return ordered
    
    def _create_cache_key(self, task_type: str, query: str, target_tokens: int, include_history_items: int) -> str:
        """Create cache key for context request"""
        # Include relevant context item IDs and timestamps for cache invalidation
        relevant_items = []
        for item_id, item in list(self.context_items.items())[-50:]:  # Last 50 items
            if item.priority.value >= ContextPriority.MEDIUM.value:
                relevant_items.append(f"{item_id}:{item.timestamp}")
        
        key_components = [
            task_type,
            query[:100],  # Truncate query for key
            str(target_tokens),
            str(include_history_items),
            "|".join(relevant_items[-20:])  # Last 20 relevant items
        ]
        
        return hashlib.md5("|".join(key_components).encode()).hexdigest()
    
    def _cleanup_old_context(self):
        """Remove old and unused context items"""
        current_time = datetime.now().timestamp()
        
        # Remove items older than 24 hours with low usage
        to_remove = []
        for item_id, item in self.context_items.items():
            age_hours = (current_time - item.timestamp) / 3600
            
            # Keep critical and high priority items longer
            if item.priority == ContextPriority.CRITICAL:
                continue  # Never remove critical
            elif item.priority == ContextPriority.HIGH and age_hours < 48:
                continue  # Keep high priority for 48 hours
            elif item.priority == ContextPriority.MEDIUM and age_hours < 24:
                continue  # Keep medium priority for 24 hours
            elif age_hours < 12:
                continue  # Keep everything for at least 12 hours
            
            # Remove if old and unused
            if item.usage_count == 0 and age_hours > 24:
                to_remove.append(item_id)
            elif item.usage_count < 2 and age_hours > 72:
                to_remove.append(item_id)
        
        for item_id in to_remove:
            del self.context_items[item_id]
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old context items")
    
    def update_context_item(self, item_id: str, content: str = None, priority: ContextPriority = None):
        """Update existing context item"""
        if item_id in self.context_items:
            item = self.context_items[item_id]
            
            if content is not None:
                item.content = content
                item.tokens = item._estimate_tokens()
                item.timestamp = datetime.now().timestamp()
            
            if priority is not None:
                item.priority = priority
    
    def remove_context_item(self, item_id: str):
        """Remove specific context item"""
        if item_id in self.context_items:
            del self.context_items[item_id]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context state"""
        total_tokens = sum(item.tokens for item in self.context_items.values())
        
        type_counts = {}
        priority_counts = {}
        
        for item in self.context_items.values():
            type_counts[item.content_type.value] = type_counts.get(item.content_type.value, 0) + 1
            priority_counts[item.priority.value] = priority_counts.get(item.priority.value, 0) + 1
        
        return {
            "total_items": len(self.context_items),
            "total_tokens": total_tokens,
            "max_context_tokens": self.max_context_tokens,
            "utilization": total_tokens / self.max_context_tokens if self.max_context_tokens > 0 else 0,
            "type_distribution": type_counts,
            "priority_distribution": priority_counts,
            "cache_hit_rate": self.cache_hits / max(1, self.total_requests),
            "total_requests": self.total_requests
        }
    
    def optimize_for_model(self, model_name: str):
        """Optimize context management for specific model"""
        self.model_name = model_name
        
        # Adjust max context tokens based on model
        model_context_limits = {
            "codellama": 16384,
            "deepseek-coder": 16384,
            "llama3": 8192,
            "qwen": 8192,
            "mistral": 8192,
            "gemma": 8192
        }
        
        for model_key, limit in model_context_limits.items():
            if model_key.lower() in model_name.lower():
                self.max_context_tokens = limit
                logger.info(f"Optimized context window for {model_name}: {limit} tokens")
                break
        
        # Clear cache when switching models
        self.context_cache.clear()
    
    def export_context_data(self) -> Dict[str, Any]:
        """Export context data for persistence"""
        return {
            "context_items": {
                item_id: {
                    "content": item.content,
                    "content_type": item.content_type.value,
                    "priority": item.priority.value,
                    "tokens": item.tokens,
                    "timestamp": item.timestamp,
                    "relevance_score": item.relevance_score,
                    "usage_count": item.usage_count,
                    "last_used": item.last_used,
                    "source": item.source,
                    "metadata": item.metadata
                }
                for item_id, item in self.context_items.items()
            },
            "max_context_tokens": self.max_context_tokens,
            "model_name": self.model_name,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits
        }
    
    def import_context_data(self, data: Dict[str, Any]):
        """Import context data from persistence"""
        if "context_items" in data:
            for item_id, item_data in data["context_items"].items():
                item = ContextItem(
                    content=item_data["content"],
                    content_type=ContentType(item_data["content_type"]),
                    priority=ContextPriority(item_data["priority"]),
                    tokens=item_data.get("tokens", 0),
                    timestamp=item_data.get("timestamp", datetime.now().timestamp()),
                    relevance_score=item_data.get("relevance_score", 1.0),
                    usage_count=item_data.get("usage_count", 0),
                    last_used=item_data.get("last_used", datetime.now().timestamp()),
                    source=item_data.get("source", "unknown"),
                    metadata=item_data.get("metadata", {})
                )
                self.context_items[item_id] = item
        
        self.max_context_tokens = data.get("max_context_tokens", 8192)
        self.model_name = data.get("model_name")
        self.total_requests = data.get("total_requests", 0)
        self.cache_hits = data.get("cache_hits", 0)
        
        logger.info(f"Imported {len(self.context_items)} context items")

# Utility functions for common context operations

def create_code_context(code_snippets: List[Dict[str, str]], manager: SmartContextManager) -> List[str]:
    """Create context from code snippets"""
    context_ids = []
    
    for snippet in code_snippets:
        code = snippet.get("code", "")
        description = snippet.get("description", "")
        language = snippet.get("language", "python")
        
        if code:
            context_id = manager.add_code_example(code, description, language)
            context_ids.append(context_id)
    
    return context_ids

def create_project_context(project_info: Dict[str, Any], manager: SmartContextManager) -> str:
    """Create project context from project intelligence"""
    context_parts = []
    
    if project_info.get("name"):
        context_parts.append(f"Project: {project_info['name']}")
    
    if project_info.get("primary_language"):
        context_parts.append(f"Primary Language: {project_info['primary_language']}")
    
    if project_info.get("frameworks"):
        frameworks = ", ".join(project_info["frameworks"])
        context_parts.append(f"Frameworks: {frameworks}")
    
    if project_info.get("project_type"):
        context_parts.append(f"Type: {project_info['project_type']}")
    
    if project_info.get("purpose"):
        context_parts.append(f"Purpose: {project_info['purpose']}")
    
    if project_info.get("key_files"):
        files = ", ".join(project_info["key_files"])
        context_parts.append(f"Key Files: {files}")
    
    project_context = "\n".join(context_parts)
    return manager.add_project_info(project_context)

def add_conversation_context(messages: List[Dict[str, str]], manager: SmartContextManager, max_messages: int = 10) -> List[str]:
    """Add conversation history to context"""
    context_ids = []
    
    # Add recent messages (most recent first, but reverse for chronological order)
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    for message in recent_messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        timestamp = message.get("timestamp")
        
        if content:
            context_id = manager.add_conversation_turn(role, content, timestamp)
            context_ids.append(context_id)
    
    return context_ids