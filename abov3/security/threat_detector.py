"""
ABOV3 Genesis - Threat Detector
Advanced threat detection and analysis system
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import hashlib
import re


class ThreatDetector:
    """
    Advanced threat detection system
    Uses behavioral analysis and pattern matching to detect threats
    """
    
    def __init__(self, audit_logger=None):
        self.audit_logger = audit_logger
        
        # Threat patterns
        self.threat_patterns = self._initialize_threat_patterns()
        
        # Behavioral tracking
        self.user_behavior: Dict[str, List[Dict[str, Any]]] = {}
        self.threat_scores: Dict[str, int] = {}
        
        # Statistics
        self.threat_stats = {
            'total_requests_analyzed': 0,
            'threats_detected': 0,
            'false_positives': 0,
            'behavioral_anomalies': 0
        }
    
    def _initialize_threat_patterns(self) -> List[Dict[str, Any]]:
        """Initialize threat detection patterns"""
        return [
            {
                'name': 'sql_injection',
                'pattern': r'(?i)(union|select|insert|update|delete|drop|exec|script)',
                'threat_score': 30,
                'category': 'injection'
            },
            {
                'name': 'xss_attempt',
                'pattern': r'(?i)(<script|javascript:|onload=|onerror=)',
                'threat_score': 25,
                'category': 'xss'
            },
            {
                'name': 'path_traversal',
                'pattern': r'(\.\./|\.\.\\\|%2e%2e)',
                'threat_score': 35,
                'category': 'path_traversal'
            },
            {
                'name': 'command_injection',
                'pattern': r'(;|&&|\|\||`|\$\()',
                'threat_score': 40,
                'category': 'command_injection'
            }
        ]
    
    async def analyze_request(self, request_data: Dict[str, Any], 
                            user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze request for threats"""
        self.threat_stats['total_requests_analyzed'] += 1
        
        result = {
            'threat_detected': False,
            'threat_score': 0,
            'threat_types': [],
            'behavioral_anomaly': False,
            'risk_factors': []
        }
        
        try:
            # Convert request to analyzable string
            request_str = str(request_data)
            user_id = user_context.get('user_id', 'anonymous') if user_context else 'anonymous'
            
            # Pattern-based threat detection
            pattern_result = await self._analyze_patterns(request_str)
            result['threat_score'] += pattern_result['score']
            result['threat_types'].extend(pattern_result['types'])
            
            # Behavioral analysis
            behavioral_result = await self._analyze_behavior(request_data, user_id)
            result['threat_score'] += behavioral_result['score']
            result['behavioral_anomaly'] = behavioral_result['anomaly']
            result['risk_factors'].extend(behavioral_result['factors'])
            
            # Determine if threat detected
            result['threat_detected'] = result['threat_score'] >= 50
            
            # Update statistics
            if result['threat_detected']:
                self.threat_stats['threats_detected'] += 1
            if result['behavioral_anomaly']:
                self.threat_stats['behavioral_anomalies'] += 1
            
            # Log threat if detected
            if result['threat_detected'] and self.audit_logger:
                await self.audit_logger.log_event("threat_detected", {
                    "threat_score": result['threat_score'],
                    "threat_types": result['threat_types'],
                    "user_id": user_id,
                    "request_summary": request_str[:200]
                }, level="SECURITY")
            
            return result
            
        except Exception as e:
            return {
                'threat_detected': True,  # Fail secure
                'threat_score': 100,
                'threat_types': ['analysis_error'],
                'error': str(e)
            }
    
    async def _analyze_patterns(self, request_str: str) -> Dict[str, Any]:
        """Analyze request against threat patterns"""
        result = {'score': 0, 'types': []}
        
        for pattern in self.threat_patterns:
            if re.search(pattern['pattern'], request_str):
                result['score'] += pattern['threat_score']
                result['types'].append(pattern['category'])
        
        return result
    
    async def _analyze_behavior(self, request_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Analyze user behavior for anomalies"""
        result = {'score': 0, 'anomaly': False, 'factors': []}
        
        # Track user behavior
        if user_id not in self.user_behavior:
            self.user_behavior[user_id] = []
        
        behavior_entry = {
            'timestamp': datetime.now().isoformat(),
            'request_type': request_data.get('type', 'unknown'),
            'request_size': len(str(request_data))
        }
        
        self.user_behavior[user_id].append(behavior_entry)
        
        # Keep only recent behavior (last 100 requests)
        self.user_behavior[user_id] = self.user_behavior[user_id][-100:]
        
        # Analyze behavior patterns
        recent_behavior = self.user_behavior[user_id][-10:]  # Last 10 requests
        
        # Check for rapid requests
        if len(recent_behavior) >= 5:
            timestamps = [datetime.fromisoformat(b['timestamp']) for b in recent_behavior]
            time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            avg_interval = sum(time_diffs) / len(time_diffs)
            
            if avg_interval < 1.0:  # Less than 1 second between requests
                result['score'] += 20
                result['anomaly'] = True
                result['factors'].append('rapid_requests')
        
        # Check for unusual request sizes
        if recent_behavior:
            sizes = [b['request_size'] for b in recent_behavior]
            avg_size = sum(sizes) / len(sizes)
            current_size = behavior_entry['request_size']
            
            if current_size > avg_size * 5:  # 5x larger than average
                result['score'] += 15
                result['factors'].append('unusual_request_size')
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get threat detection statistics"""
        return self.threat_stats.copy()