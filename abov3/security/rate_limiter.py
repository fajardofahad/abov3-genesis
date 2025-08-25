"""
ABOV3 Genesis - Rate Limiter and DDoS Protection
Advanced rate limiting and DDoS protection system
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum


class LimitType(Enum):
    """Types of rate limits"""
    REQUESTS_PER_SECOND = "requests_per_second"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    CONCURRENT_CONNECTIONS = "concurrent_connections"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    limit_type: LimitType
    limit: int
    window_size: int
    burst_limit: Optional[int] = None


class RateLimiter:
    """Advanced rate limiter with multiple algorithms"""
    
    def __init__(self, max_requests: int = 1000, audit_logger=None):
        self.max_requests = max_requests
        self.audit_logger = audit_logger
        
        # Client tracking
        self.client_requests: Dict[str, deque] = defaultdict(deque)
        self.client_limits: Dict[str, List[RateLimit]] = {}
        self.blocked_clients: Dict[str, datetime] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'unique_clients': 0,
            'blocked_clients_count': 0
        }
        
        # Default rate limits
        self.default_limits = [
            RateLimit(LimitType.REQUESTS_PER_SECOND, 10, 1),
            RateLimit(LimitType.REQUESTS_PER_MINUTE, 100, 60),
            RateLimit(LimitType.REQUESTS_PER_HOUR, 1000, 3600)
        ]
    
    async def check_rate_limit(self, client_id: str, request_weight: int = 1) -> bool:
        """Check if client is within rate limits"""
        self.stats['total_requests'] += 1
        
        current_time = time.time()
        
        # Check if client is blocked
        if client_id in self.blocked_clients:
            if datetime.now() < self.blocked_clients[client_id]:
                self.stats['blocked_requests'] += 1
                return False
            else:
                del self.blocked_clients[client_id]
        
        # Track unique clients
        if client_id not in self.client_requests:
            self.stats['unique_clients'] += 1
        
        # Get client-specific limits or use defaults
        limits = self.client_limits.get(client_id, self.default_limits)
        
        # Check each rate limit
        for rate_limit in limits:
            if not await self._check_limit(client_id, rate_limit, current_time, request_weight):
                self.stats['blocked_requests'] += 1
                await self._handle_rate_limit_exceeded(client_id, rate_limit)
                return False
        
        # Record successful request
        self.client_requests[client_id].append((current_time, request_weight))
        self._cleanup_old_requests(client_id, current_time)
        
        return True
    
    async def _check_limit(self, client_id: str, rate_limit: RateLimit, 
                          current_time: float, request_weight: int) -> bool:
        """Check specific rate limit"""
        requests = self.client_requests[client_id]
        window_start = current_time - rate_limit.window_size
        
        # Count requests in window
        request_count = sum(
            weight for timestamp, weight in requests 
            if timestamp >= window_start
        )
        
        return (request_count + request_weight) <= rate_limit.limit
    
    async def _handle_rate_limit_exceeded(self, client_id: str, rate_limit: RateLimit):
        """Handle rate limit exceeded event"""
        # Log rate limit exceeded
        if self.audit_logger:
            await self.audit_logger.log_event("rate_limit_exceeded", {
                "client_id": client_id,
                "limit_type": rate_limit.limit_type.value,
                "limit": rate_limit.limit,
                "window_size": rate_limit.window_size
            })
        
        # Implement progressive penalties
        violation_count = getattr(self, f'_violations_{client_id}', 0) + 1
        setattr(self, f'_violations_{client_id}', violation_count)
        
        if violation_count >= 5:  # Block after 5 violations
            block_duration = min(300 * (violation_count - 4), 3600)  # Max 1 hour
            self.blocked_clients[client_id] = datetime.now() + timedelta(seconds=block_duration)
            self.stats['blocked_clients_count'] += 1
    
    def _cleanup_old_requests(self, client_id: str, current_time: float):
        """Remove old requests outside all windows"""
        requests = self.client_requests[client_id]
        max_window = max(limit.window_size for limit in self.default_limits)
        cutoff = current_time - max_window
        
        while requests and requests[0][0] < cutoff:
            requests.popleft()
    
    async def emergency_block_all(self):
        """Emergency block all new requests"""
        current_time = datetime.now()
        for client_id in list(self.client_requests.keys()):
            self.blocked_clients[client_id] = current_time + timedelta(hours=1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return self.stats.copy()


class DDosProtection:
    """Advanced DDoS protection system"""
    
    def __init__(self, audit_logger=None):
        self.audit_logger = audit_logger
        
        # Attack detection
        self.request_patterns: Dict[str, List[float]] = defaultdict(list)
        self.suspicious_ips: Set[str] = set()
        self.attack_signatures: Dict[str, int] = defaultdict(int)
        
        # Thresholds
        self.burst_threshold = 50  # requests per second
        self.sustained_threshold = 200  # requests per minute
        self.geographic_threshold = 10  # requests from same region
        
        # Statistics
        self.ddos_stats = {
            'attacks_detected': 0,
            'blocked_ips': 0,
            'suspicious_patterns': 0
        }
    
    async def is_attack_detected(self, client_ip: str) -> bool:
        """Detect if client IP is part of DDoS attack"""
        current_time = time.time()
        
        # Record request
        self.request_patterns[client_ip].append(current_time)
        
        # Cleanup old requests (keep last hour)
        cutoff = current_time - 3600
        self.request_patterns[client_ip] = [
            t for t in self.request_patterns[client_ip] if t >= cutoff
        ]
        
        # Check for attack patterns
        if await self._detect_burst_attack(client_ip, current_time):
            await self._handle_attack_detected(client_ip, "burst_attack")
            return True
        
        if await self._detect_sustained_attack(client_ip, current_time):
            await self._handle_attack_detected(client_ip, "sustained_attack")
            return True
        
        return client_ip in self.suspicious_ips
    
    async def _detect_burst_attack(self, client_ip: str, current_time: float) -> bool:
        """Detect burst attack (many requests in short time)"""
        recent_requests = [
            t for t in self.request_patterns[client_ip] 
            if t >= current_time - 1  # Last second
        ]
        return len(recent_requests) > self.burst_threshold
    
    async def _detect_sustained_attack(self, client_ip: str, current_time: float) -> bool:
        """Detect sustained attack (high request rate over time)"""
        recent_requests = [
            t for t in self.request_patterns[client_ip] 
            if t >= current_time - 60  # Last minute
        ]
        return len(recent_requests) > self.sustained_threshold
    
    async def _handle_attack_detected(self, client_ip: str, attack_type: str):
        """Handle detected DDoS attack"""
        self.suspicious_ips.add(client_ip)
        self.attack_signatures[attack_type] += 1
        self.ddos_stats['attacks_detected'] += 1
        self.ddos_stats['blocked_ips'] += 1
        
        if self.audit_logger:
            await self.audit_logger.log_event("ddos_attack_detected", {
                "client_ip": client_ip,
                "attack_type": attack_type,
                "timestamp": datetime.now().isoformat()
            }, level="SECURITY")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get DDoS protection statistics"""
        return self.ddos_stats.copy()