"""
ABOV3 Genesis - Debug Audit Logger
Comprehensive audit logging system for debug operations with enterprise compliance
"""

import asyncio
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import gzip
from collections import defaultdict
import threading

from .crypto_manager import CryptographyManager


class AuditEventType(Enum):
    """Types of audit events"""
    # Session events
    SESSION_CREATED = "session_created"
    SESSION_TERMINATED = "session_terminated"
    SESSION_SUSPENDED = "session_suspended"
    SESSION_EXPIRED = "session_expired"
    
    # Authentication events
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    MFA_VERIFICATION = "mfa_verification"
    TOKEN_VALIDATION = "token_validation"
    
    # Authorization events
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_ASSIGNED = "role_assigned"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    
    # Debug operations
    DEBUG_START = "debug_start"
    DEBUG_STOP = "debug_stop"
    BREAKPOINT_SET = "breakpoint_set"
    BREAKPOINT_HIT = "breakpoint_hit"
    VARIABLE_READ = "variable_read"
    VARIABLE_MODIFIED = "variable_modified"
    CODE_EXECUTION = "code_execution"
    MEMORY_ACCESS = "memory_access"
    FILE_ACCESS = "file_access"
    
    # Data events
    DATA_STORED = "data_stored"
    DATA_ACCESSED = "data_accessed"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    
    # Security events
    SECURITY_VIOLATION = "security_violation"
    THREAT_DETECTED = "threat_detected"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGED = "configuration_changed"
    ERROR_OCCURRED = "error_occurred"


class AuditSeverity(Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStandard(Enum):
    """Compliance standards for audit events"""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    NIST = "nist"
    ISO27001 = "iso27001"


@dataclass
class AuditEvent:
    """Individual audit event record"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    client_ip: Optional[str]
    user_agent: Optional[str]
    
    # Event details
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    severity: AuditSeverity = AuditSeverity.MEDIUM
    
    # Context and metadata
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    
    # Security classification
    sensitive_data: bool = False
    classification_level: str = "internal"
    
    # Additional tracking
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Risk assessment
    risk_score: int = 0
    threat_indicators: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        
        # Auto-assign compliance standards based on event type
        self._assign_compliance_standards()
        
        # Calculate risk score if not provided
        if self.risk_score == 0:
            self.risk_score = self._calculate_risk_score()
    
    def _assign_compliance_standards(self):
        """Automatically assign compliance standards based on event type"""
        # SOC 2 - Security, Availability, Processing Integrity
        if self.event_type in [
            AuditEventType.AUTH_SUCCESS, AuditEventType.AUTH_FAILURE,
            AuditEventType.PERMISSION_DENIED, AuditEventType.SECURITY_VIOLATION,
            AuditEventType.SESSION_CREATED, AuditEventType.SESSION_TERMINATED,
            AuditEventType.DATA_ACCESSED, AuditEventType.DATA_MODIFIED
        ]:
            if ComplianceStandard.SOC2 not in self.compliance_standards:
                self.compliance_standards.append(ComplianceStandard.SOC2)
        
        # GDPR - Data protection and privacy
        if self.event_type in [
            AuditEventType.DATA_ACCESSED, AuditEventType.DATA_MODIFIED,
            AuditEventType.DATA_DELETED, AuditEventType.DATA_EXPORTED
        ] and self.sensitive_data:
            if ComplianceStandard.GDPR not in self.compliance_standards:
                self.compliance_standards.append(ComplianceStandard.GDPR)
        
        # NIST - Security framework
        if self.event_type in [
            AuditEventType.SECURITY_VIOLATION, AuditEventType.THREAT_DETECTED,
            AuditEventType.INTRUSION_ATTEMPT, AuditEventType.SUSPICIOUS_ACTIVITY
        ]:
            if ComplianceStandard.NIST not in self.compliance_standards:
                self.compliance_standards.append(ComplianceStandard.NIST)
    
    def _calculate_risk_score(self) -> int:
        """Calculate risk score based on event characteristics"""
        score = 0
        
        # Base score by event type
        high_risk_events = [
            AuditEventType.SECURITY_VIOLATION, AuditEventType.THREAT_DETECTED,
            AuditEventType.PRIVILEGE_ESCALATION, AuditEventType.INTRUSION_ATTEMPT
        ]
        
        if self.event_type in high_risk_events:
            score += 50
        elif self.severity == AuditSeverity.CRITICAL:
            score += 40
        elif self.severity == AuditSeverity.HIGH:
            score += 30
        elif self.severity == AuditSeverity.MEDIUM:
            score += 15
        else:
            score += 5
        
        # Additional factors
        if self.sensitive_data:
            score += 20
        
        if len(self.threat_indicators) > 0:
            score += len(self.threat_indicators) * 10
        
        if self.result == "failure":
            score += 10
        
        return min(score, 100)  # Cap at 100


@dataclass
class AuditConfiguration:
    """Audit system configuration"""
    # Storage settings
    log_retention_days: int = 2555  # 7 years (compliance requirement)
    max_log_size_gb: int = 50
    compression_enabled: bool = True
    encryption_enabled: bool = True
    
    # Logging levels
    min_severity: AuditSeverity = AuditSeverity.LOW
    log_all_data_access: bool = True
    log_all_auth_events: bool = True
    log_failed_attempts: bool = True
    
    # Real-time monitoring
    real_time_alerts: bool = True
    alert_threshold_score: int = 75
    max_events_per_second: int = 1000
    
    # Compliance settings
    enabled_standards: List[ComplianceStandard] = field(default_factory=lambda: [
        ComplianceStandard.SOC2,
        ComplianceStandard.GDPR,
        ComplianceStandard.NIST
    ])
    
    # Export settings
    enable_siem_export: bool = True
    siem_export_format: str = "json"
    export_batch_size: int = 1000


class DebugAuditLogger:
    """
    Enterprise-grade audit logging system for debug operations
    Provides comprehensive logging, compliance reporting, and security monitoring
    """
    
    def __init__(
        self,
        audit_dir: Path,
        crypto_manager: Optional[CryptographyManager] = None,
        config: Optional[AuditConfiguration] = None
    ):
        self.audit_dir = audit_dir
        self.crypto_manager = crypto_manager
        self.config = config or AuditConfiguration()
        
        # Create audit directory structure
        self.logs_dir = audit_dir / "logs"
        self.archive_dir = audit_dir / "archive"
        self.exports_dir = audit_dir / "exports"
        self.temp_dir = audit_dir / "temp"
        
        for directory in [self.logs_dir, self.archive_dir, self.exports_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Current log file tracking
        self.current_log_file = None
        self.current_log_size = 0
        self.max_log_file_size = 100 * 1024 * 1024  # 100MB per file
        
        # In-memory event buffer for high-performance logging
        self.event_buffer: List[AuditEvent] = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 1000
        self.buffer_flush_interval = 30  # seconds
        
        # Event tracking and analytics
        self.event_counts: Dict[AuditEventType, int] = defaultdict(int)
        self.user_activity: Dict[str, List[datetime]] = defaultdict(list)
        self.ip_activity: Dict[str, List[datetime]] = defaultdict(list)
        self.threat_indicators: Dict[str, int] = defaultdict(int)
        
        # Real-time monitoring
        self.alert_handlers: List[callable] = []
        self.monitoring_enabled = True
        
        # Setup logging
        self.logger = logging.getLogger('abov3.security.debug_audit')
        self._setup_file_logger()
        
        # Background tasks
        self._flush_task = None
        self._cleanup_task = None
        self._monitoring_task = None
        
        # Start background operations
        self._start_background_tasks()
    
    async def log_event(
        self,
        event_type: Union[AuditEventType, str],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        sensitive_data: bool = False,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log an audit event with comprehensive metadata
        
        Args:
            event_type: Type of event being logged
            user_id: User identifier
            session_id: Session identifier
            client_ip: Client IP address
            user_agent: Client user agent
            resource: Resource being accessed
            action: Action being performed
            result: Result of the action
            severity: Event severity level
            context: Additional context data
            tags: Event tags
            sensitive_data: Whether event contains sensitive data
            correlation_id: Correlation ID for event grouping
            **kwargs: Additional event data
            
        Returns:
            str: Event ID
        """
        try:
            # Convert string event type to enum if needed
            if isinstance(event_type, str):
                try:
                    event_type = AuditEventType(event_type)
                except ValueError:
                    # Create a custom event type marker
                    self.logger.warning(f"Unknown event type: {event_type}")
                    event_type = AuditEventType.ERROR_OCCURRED
            
            # Create audit event
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id,
                client_ip=client_ip,
                user_agent=user_agent,
                resource=resource,
                action=action,
                result=result,
                severity=severity,
                context=context or {},
                tags=tags or [],
                sensitive_data=sensitive_data,
                correlation_id=correlation_id
            )
            
            # Add any additional context from kwargs
            event.context.update(kwargs)
            
            # Check severity threshold
            if self._should_log_event(event):
                # Add to buffer for batch processing
                with self.buffer_lock:
                    self.event_buffer.append(event)
                    
                    # Flush buffer if it's getting full
                    if len(self.event_buffer) >= self.max_buffer_size:
                        await self._flush_event_buffer()
                
                # Update analytics
                self._update_analytics(event)
                
                # Real-time monitoring and alerts
                if self.monitoring_enabled:
                    await self._process_real_time_monitoring(event)
                
                self.logger.debug(f"Audit event logged: {event.event_type.value} ({event.event_id})")
            
            return event.event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
            # Fallback logging to ensure we don't lose critical events
            self.logger.critical(f"AUDIT_FAILURE: {event_type} - {str(e)}")
            return ""
    
    async def log_debug_operation(
        self,
        operation: str,
        user_id: str,
        session_id: str,
        target_resource: Optional[str] = None,
        operation_result: Optional[str] = None,
        data_accessed: Optional[Dict[str, Any]] = None,
        modifications_made: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Log a debug-specific operation with detailed context
        
        Args:
            operation: Debug operation being performed
            user_id: User performing the operation
            session_id: Debug session identifier
            target_resource: Target resource (file, variable, etc.)
            operation_result: Result of the operation
            data_accessed: Data that was accessed
            modifications_made: Modifications that were made
            **kwargs: Additional context
            
        Returns:
            str: Event ID
        """
        # Determine event type based on operation
        event_type_mapping = {
            'breakpoint_set': AuditEventType.BREAKPOINT_SET,
            'breakpoint_hit': AuditEventType.BREAKPOINT_HIT,
            'variable_read': AuditEventType.VARIABLE_READ,
            'variable_modify': AuditEventType.VARIABLE_MODIFIED,
            'code_execute': AuditEventType.CODE_EXECUTION,
            'memory_access': AuditEventType.MEMORY_ACCESS,
            'file_access': AuditEventType.FILE_ACCESS,
            'debug_start': AuditEventType.DEBUG_START,
            'debug_stop': AuditEventType.DEBUG_STOP
        }
        
        event_type = event_type_mapping.get(operation, AuditEventType.DEBUG_START)
        
        # Determine severity
        severity = AuditSeverity.MEDIUM
        if modifications_made:
            severity = AuditSeverity.HIGH
        elif operation in ['code_execute', 'memory_access']:
            severity = AuditSeverity.HIGH
        
        # Check for sensitive data
        sensitive_data = False
        if data_accessed:
            sensitive_keywords = ['password', 'key', 'secret', 'token', 'credential']
            data_str = json.dumps(data_accessed).lower()
            sensitive_data = any(keyword in data_str for keyword in sensitive_keywords)
        
        context = {
            'debug_operation': operation,
            'target_resource': target_resource,
            'operation_result': operation_result,
            'data_accessed_summary': self._summarize_data_access(data_accessed),
            'modifications_summary': self._summarize_modifications(modifications_made),
            **kwargs
        }
        
        return await self.log_event(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            resource=target_resource,
            action=operation,
            result=operation_result,
            severity=severity,
            context=context,
            sensitive_data=sensitive_data,
            tags=['debug', 'operation']
        )
    
    async def log_security_event(
        self,
        security_event_type: str,
        user_id: Optional[str],
        session_id: Optional[str],
        threat_level: str = "medium",
        threat_indicators: Optional[List[str]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Log a security-related event with threat intelligence
        
        Args:
            security_event_type: Type of security event
            user_id: User involved in the event
            session_id: Session involved in the event
            threat_level: Level of threat (low, medium, high, critical)
            threat_indicators: List of threat indicators
            additional_context: Additional security context
            **kwargs: Additional data
            
        Returns:
            str: Event ID
        """
        # Map security event types
        security_event_mapping = {
            'violation': AuditEventType.SECURITY_VIOLATION,
            'threat_detected': AuditEventType.THREAT_DETECTED,
            'suspicious_activity': AuditEventType.SUSPICIOUS_ACTIVITY,
            'intrusion_attempt': AuditEventType.INTRUSION_ATTEMPT
        }
        
        event_type = security_event_mapping.get(security_event_type, AuditEventType.SECURITY_VIOLATION)
        
        # Map threat level to severity
        severity_mapping = {
            'low': AuditSeverity.LOW,
            'medium': AuditSeverity.MEDIUM,
            'high': AuditSeverity.HIGH,
            'critical': AuditSeverity.CRITICAL
        }
        
        severity = severity_mapping.get(threat_level, AuditSeverity.MEDIUM)
        
        context = {
            'security_event_type': security_event_type,
            'threat_level': threat_level,
            'threat_indicators': threat_indicators or [],
            **(additional_context or {}),
            **kwargs
        }
        
        return await self.log_event(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            severity=severity,
            context=context,
            tags=['security', 'threat'],
            sensitive_data=True
        )
    
    async def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_types: Optional[List[AuditEventType]] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 1000,
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query audit events with filtering
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            user_id: User filter
            session_id: Session filter
            event_types: Event type filter
            severity: Minimum severity filter
            limit: Maximum number of results
            include_context: Whether to include full context
            
        Returns:
            List of matching events
        """
        try:
            # For now, we'll search through current buffer and recent log files
            # In production, this would use a proper database or search index
            
            events = []
            
            # Search buffer first
            with self.buffer_lock:
                for event in self.event_buffer:
                    if self._event_matches_query(
                        event, start_time, end_time, user_id, session_id, event_types, severity
                    ):
                        events.append(self._serialize_event(event, include_context))
            
            # Search recent log files
            # This is a simplified implementation - production would use proper indexing
            for log_file in sorted(self.logs_dir.glob("audit_*.json"), reverse=True):
                if len(events) >= limit:
                    break
                    
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            if len(events) >= limit:
                                break
                            
                            try:
                                event_data = json.loads(line.strip())
                                event = self._deserialize_event(event_data)
                                
                                if self._event_matches_query(
                                    event, start_time, end_time, user_id, session_id, event_types, severity
                                ):
                                    events.append(self._serialize_event(event, include_context))
                            except:
                                continue
                except Exception as e:
                    self.logger.warning(f"Error reading log file {log_file}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return events[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to query events: {e}")
            return []
    
    async def generate_compliance_report(
        self,
        standard: ComplianceStandard,
        start_date: datetime,
        end_date: datetime,
        user_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report for specific standard
        
        Args:
            standard: Compliance standard to report on
            start_date: Report start date
            end_date: Report end date
            user_filter: Optional user filter
            
        Returns:
            Dict containing compliance report data
        """
        try:
            # Query relevant events
            all_events = await self.query_events(
                start_time=start_date,
                end_time=end_date,
                user_id=user_filter,
                limit=10000,
                include_context=True
            )
            
            # Filter events for the specific compliance standard
            relevant_events = [
                event for event in all_events
                if standard.value in event.get('compliance_standards', [])
            ]
            
            # Generate report based on standard
            if standard == ComplianceStandard.SOC2:
                return await self._generate_soc2_report(relevant_events, start_date, end_date)
            elif standard == ComplianceStandard.GDPR:
                return await self._generate_gdpr_report(relevant_events, start_date, end_date)
            elif standard == ComplianceStandard.NIST:
                return await self._generate_nist_report(relevant_events, start_date, end_date)
            else:
                return await self._generate_generic_report(relevant_events, start_date, end_date, standard)
                
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            return {
                'error': str(e),
                'standard': standard.value,
                'period': f"{start_date} to {end_date}"
            }
    
    async def export_events(
        self,
        format_type: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        compress: bool = True
    ) -> Optional[Path]:
        """
        Export audit events to file
        
        Args:
            format_type: Export format (json, csv, xml)
            start_time: Start time filter
            end_time: End time filter
            compress: Whether to compress the export
            
        Returns:
            Path to exported file, or None if failed
        """
        try:
            # Query events
            events = await self.query_events(
                start_time=start_time,
                end_time=end_time,
                limit=100000,
                include_context=True
            )
            
            if not events:
                return None
            
            # Generate export filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_export_{timestamp}.{format_type}"
            if compress:
                filename += ".gz"
            
            export_path = self.exports_dir / filename
            
            # Export based on format
            if format_type.lower() == "json":
                await self._export_json(events, export_path, compress)
            elif format_type.lower() == "csv":
                await self._export_csv(events, export_path, compress)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            # Log export event
            await self.log_event(
                event_type=AuditEventType.DATA_EXPORTED,
                action="audit_export",
                result="success",
                context={
                    "export_format": format_type,
                    "event_count": len(events),
                    "export_file": str(export_path),
                    "compressed": compress
                }
            )
            
            return export_path
            
        except Exception as e:
            self.logger.error(f"Failed to export events: {e}")
            return None
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get comprehensive audit statistics"""
        return {
            'event_counts': dict(self.event_counts),
            'total_events': sum(self.event_counts.values()),
            'buffer_size': len(self.event_buffer),
            'unique_users': len(self.user_activity),
            'unique_ips': len(self.ip_activity),
            'threat_indicators': dict(self.threat_indicators),
            'monitoring_enabled': self.monitoring_enabled,
            'current_log_size': self.current_log_size,
            'retention_period_days': self.config.log_retention_days,
            'compliance_standards': [std.value for std in self.config.enabled_standards]
        }
    
    def add_alert_handler(self, handler: callable):
        """Add a real-time alert handler"""
        self.alert_handlers.append(handler)
    
    def _should_log_event(self, event: AuditEvent) -> bool:
        """Determine if event should be logged based on configuration"""
        # Check severity threshold
        severity_levels = {
            AuditSeverity.LOW: 1,
            AuditSeverity.MEDIUM: 2,
            AuditSeverity.HIGH: 3,
            AuditSeverity.CRITICAL: 4
        }
        
        min_level = severity_levels.get(self.config.min_severity, 1)
        event_level = severity_levels.get(event.severity, 1)
        
        return event_level >= min_level
    
    def _update_analytics(self, event: AuditEvent):
        """Update real-time analytics"""
        # Update event counts
        self.event_counts[event.event_type] += 1
        
        # Update user activity
        if event.user_id:
            self.user_activity[event.user_id].append(event.timestamp)
            # Keep only recent activity (last 24 hours)
            cutoff = datetime.now() - timedelta(hours=24)
            self.user_activity[event.user_id] = [
                ts for ts in self.user_activity[event.user_id] if ts > cutoff
            ]
        
        # Update IP activity
        if event.client_ip:
            self.ip_activity[event.client_ip].append(event.timestamp)
            cutoff = datetime.now() - timedelta(hours=24)
            self.ip_activity[event.client_ip] = [
                ts for ts in self.ip_activity[event.client_ip] if ts > cutoff
            ]
        
        # Update threat indicators
        for indicator in event.threat_indicators:
            self.threat_indicators[indicator] += 1
    
    async def _process_real_time_monitoring(self, event: AuditEvent):
        """Process event for real-time monitoring and alerting"""
        # Check for high-risk events
        if event.risk_score >= self.config.alert_threshold_score or event.severity == AuditSeverity.CRITICAL:
            await self._trigger_alert(event)
        
        # Check for suspicious patterns
        if event.user_id:
            recent_events = self.user_activity.get(event.user_id, [])
            if len(recent_events) > 100:  # More than 100 events in 24h
                await self._trigger_alert(event, "Suspicious user activity volume")
        
        if event.client_ip:
            recent_events = self.ip_activity.get(event.client_ip, [])
            if len(recent_events) > 500:  # More than 500 events from same IP in 24h
                await self._trigger_alert(event, "Suspicious IP activity volume")
    
    async def _trigger_alert(self, event: AuditEvent, reason: str = "High risk event"):
        """Trigger security alert"""
        alert_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'user_id': event.user_id,
            'client_ip': event.client_ip,
            'severity': event.severity.value,
            'risk_score': event.risk_score,
            'reason': reason,
            'timestamp': event.timestamp.isoformat(),
            'threat_indicators': event.threat_indicators
        }
        
        # Call all registered alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert_data)
                else:
                    handler(alert_data)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}")
        
        # Log the alert itself
        await self.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.HIGH,
            context=alert_data,
            tags=['alert', 'security']
        )
    
    async def _flush_event_buffer(self):
        """Flush event buffer to persistent storage"""
        if not self.event_buffer:
            return
        
        try:
            with self.buffer_lock:
                events_to_flush = self.event_buffer.copy()
                self.event_buffer.clear()
            
            # Ensure we have a current log file
            if not self.current_log_file or self.current_log_size >= self.max_log_file_size:
                self._rotate_log_file()
            
            # Write events to file
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                for event in events_to_flush:
                    event_data = self._serialize_event(event, include_context=True)
                    
                    # Encrypt if enabled
                    if self.config.encryption_enabled and self.crypto_manager:
                        event_json = json.dumps(event_data, ensure_ascii=False)
                        encrypted_data = await self.crypto_manager.encrypt_data(event_json.encode())
                        # Store as base64 for JSON compatibility
                        import base64
                        event_data = {
                            'encrypted': True,
                            'data': base64.b64encode(encrypted_data).decode()
                        }
                    
                    line = json.dumps(event_data, ensure_ascii=False) + '\n'
                    f.write(line)
                    self.current_log_size += len(line.encode())
            
            self.logger.debug(f"Flushed {len(events_to_flush)} events to {self.current_log_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to flush event buffer: {e}")
    
    def _rotate_log_file(self):
        """Rotate to a new log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = self.logs_dir / f"audit_{timestamp}.json"
        self.current_log_size = 0
        
        self.logger.info(f"Rotated to new log file: {self.current_log_file}")
    
    def _serialize_event(self, event: AuditEvent, include_context: bool = True) -> Dict[str, Any]:
        """Serialize audit event to dictionary"""
        data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'user_id': event.user_id,
            'session_id': event.session_id,
            'client_ip': event.client_ip,
            'user_agent': event.user_agent,
            'resource': event.resource,
            'action': event.action,
            'result': event.result,
            'severity': event.severity.value,
            'classification_level': event.classification_level,
            'sensitive_data': event.sensitive_data,
            'risk_score': event.risk_score,
            'compliance_standards': [std.value for std in event.compliance_standards],
            'tags': event.tags,
            'correlation_id': event.correlation_id,
            'parent_event_id': event.parent_event_id,
            'request_id': event.request_id,
            'threat_indicators': event.threat_indicators
        }
        
        if include_context:
            data['context'] = event.context
        
        return data
    
    def _deserialize_event(self, data: Dict[str, Any]) -> AuditEvent:
        """Deserialize audit event from dictionary"""
        return AuditEvent(
            event_id=data['event_id'],
            event_type=AuditEventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            client_ip=data.get('client_ip'),
            user_agent=data.get('user_agent'),
            resource=data.get('resource'),
            action=data.get('action'),
            result=data.get('result'),
            severity=AuditSeverity(data.get('severity', 'medium')),
            classification_level=data.get('classification_level', 'internal'),
            sensitive_data=data.get('sensitive_data', False),
            risk_score=data.get('risk_score', 0),
            compliance_standards=[ComplianceStandard(std) for std in data.get('compliance_standards', [])],
            tags=data.get('tags', []),
            context=data.get('context', {}),
            correlation_id=data.get('correlation_id'),
            parent_event_id=data.get('parent_event_id'),
            request_id=data.get('request_id'),
            threat_indicators=data.get('threat_indicators', [])
        )
    
    def _event_matches_query(
        self,
        event: AuditEvent,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        user_id: Optional[str],
        session_id: Optional[str],
        event_types: Optional[List[AuditEventType]],
        severity: Optional[AuditSeverity]
    ) -> bool:
        """Check if event matches query criteria"""
        if start_time and event.timestamp < start_time:
            return False
        if end_time and event.timestamp > end_time:
            return False
        if user_id and event.user_id != user_id:
            return False
        if session_id and event.session_id != session_id:
            return False
        if event_types and event.event_type not in event_types:
            return False
        
        if severity:
            severity_levels = {
                AuditSeverity.LOW: 1,
                AuditSeverity.MEDIUM: 2,
                AuditSeverity.HIGH: 3,
                AuditSeverity.CRITICAL: 4
            }
            min_level = severity_levels.get(severity, 1)
            event_level = severity_levels.get(event.severity, 1)
            if event_level < min_level:
                return False
        
        return True
    
    def _summarize_data_access(self, data_accessed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Create a summary of accessed data for audit purposes"""
        if not data_accessed:
            return None
        
        return {
            'data_types': list(data_accessed.keys()),
            'total_fields': sum(len(v) if isinstance(v, (list, dict)) else 1 for v in data_accessed.values()),
            'sensitive_detected': any(
                keyword in str(data_accessed).lower() 
                for keyword in ['password', 'secret', 'key', 'token']
            )
        }
    
    def _summarize_modifications(self, modifications: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Create a summary of modifications for audit purposes"""
        if not modifications:
            return None
        
        return {
            'modified_fields': list(modifications.keys()),
            'modification_count': len(modifications),
            'modification_types': list(set(type(v).__name__ for v in modifications.values()))
        }
    
    async def _generate_soc2_report(self, events: List[Dict[str, Any]], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate SOC 2 compliance report"""
        # SOC 2 focuses on Security, Availability, Processing Integrity, Confidentiality, Privacy
        
        auth_events = [e for e in events if 'auth' in e['event_type']]
        access_events = [e for e in events if 'access' in e['event_type']]
        security_events = [e for e in events if e['severity'] in ['high', 'critical']]
        
        return {
            'report_type': 'SOC 2',
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'summary': {
                'total_events': len(events),
                'authentication_events': len(auth_events),
                'access_events': len(access_events),
                'security_incidents': len(security_events)
            },
            'security_controls': {
                'authentication_monitoring': len(auth_events) > 0,
                'access_logging': len(access_events) > 0,
                'incident_tracking': len(security_events) > 0
            },
            'recommendations': self._generate_soc2_recommendations(events)
        }
    
    async def _generate_gdpr_report(self, events: List[Dict[str, Any]], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        # GDPR focuses on personal data protection
        
        data_access_events = [e for e in events if e.get('sensitive_data', False)]
        data_deletion_events = [e for e in events if e['event_type'] == 'data_deleted']
        data_export_events = [e for e in events if e['event_type'] == 'data_exported']
        
        return {
            'report_type': 'GDPR',
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'summary': {
                'total_events': len(events),
                'personal_data_access': len(data_access_events),
                'data_deletions': len(data_deletion_events),
                'data_exports': len(data_export_events)
            },
            'data_subject_rights': {
                'right_to_access_supported': len(data_export_events) > 0,
                'right_to_erasure_supported': len(data_deletion_events) > 0,
                'right_to_portability_supported': len(data_export_events) > 0
            },
            'recommendations': self._generate_gdpr_recommendations(events)
        }
    
    async def _generate_nist_report(self, events: List[Dict[str, Any]], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate NIST framework compliance report"""
        # NIST focuses on Identify, Protect, Detect, Respond, Recover
        
        threat_events = [e for e in events if 'threat' in e['event_type'] or 'security' in e['event_type']]
        incident_responses = [e for e in events if e.get('result') == 'blocked' or 'suspended' in e.get('action', '')]
        
        return {
            'report_type': 'NIST Cybersecurity Framework',
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'framework_functions': {
                'identify': {
                    'asset_monitoring': len([e for e in events if 'access' in e['event_type']]) > 0,
                    'vulnerability_management': len([e for e in events if 'vulnerability' in str(e.get('context', {}))]) > 0
                },
                'protect': {
                    'access_control': len([e for e in events if 'permission' in e['event_type']]) > 0,
                    'data_security': len([e for e in events if e.get('sensitive_data', False)]) > 0
                },
                'detect': {
                    'anomaly_detection': len(threat_events) > 0,
                    'security_monitoring': len(events) > 0
                },
                'respond': {
                    'incident_response': len(incident_responses) > 0,
                    'communication': True  # Audit logging itself provides communication
                },
                'recover': {
                    'recovery_planning': len([e for e in events if 'recovery' in str(e.get('context', {}))]) > 0
                }
            },
            'threat_landscape': {
                'total_threats_detected': len(threat_events),
                'threat_types': list(set(e['event_type'] for e in threat_events))
            },
            'recommendations': self._generate_nist_recommendations(events)
        }
    
    async def _generate_generic_report(self, events: List[Dict[str, Any]], start_date: datetime, end_date: datetime, standard: ComplianceStandard) -> Dict[str, Any]:
        """Generate generic compliance report"""
        return {
            'report_type': standard.value.upper(),
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'summary': {
                'total_events': len(events),
                'event_types': list(set(e['event_type'] for e in events)),
                'unique_users': len(set(e['user_id'] for e in events if e.get('user_id'))),
                'severity_distribution': {
                    severity: len([e for e in events if e.get('severity') == severity])
                    for severity in ['low', 'medium', 'high', 'critical']
                }
            },
            'recommendations': []
        }
    
    def _generate_soc2_recommendations(self, events: List[Dict[str, Any]]) -> List[str]:
        """Generate SOC 2 specific recommendations"""
        recommendations = []
        
        security_events = [e for e in events if e.get('severity') in ['high', 'critical']]
        if len(security_events) > 10:
            recommendations.append("High number of security events detected. Review and strengthen security controls.")
        
        failed_auth = [e for e in events if e['event_type'] == 'auth_failure']
        if len(failed_auth) > 50:
            recommendations.append("Excessive authentication failures detected. Consider implementing additional security measures.")
        
        return recommendations
    
    def _generate_gdpr_recommendations(self, events: List[Dict[str, Any]]) -> List[str]:
        """Generate GDPR specific recommendations"""
        recommendations = []
        
        sensitive_access = [e for e in events if e.get('sensitive_data', False)]
        if len(sensitive_access) > 100:
            recommendations.append("High volume of personal data access. Ensure data minimization principles are applied.")
        
        if not any(e['event_type'] == 'data_deleted' for e in events):
            recommendations.append("No data deletion events recorded. Ensure right to erasure mechanisms are in place.")
        
        return recommendations
    
    def _generate_nist_recommendations(self, events: List[Dict[str, Any]]) -> List[str]:
        """Generate NIST framework specific recommendations"""
        recommendations = []
        
        threat_events = [e for e in events if 'threat' in e['event_type'] or 'security' in e['event_type']]
        if len(threat_events) == 0:
            recommendations.append("No threat detection events found. Verify threat detection capabilities are functioning.")
        
        if not any('respond' in str(e.get('action', '')) for e in events):
            recommendations.append("Limited incident response activity recorded. Review incident response procedures.")
        
        return recommendations
    
    async def _export_json(self, events: List[Dict[str, Any]], export_path: Path, compress: bool):
        """Export events to JSON format"""
        export_data = {
            'export_metadata': {
                'timestamp': datetime.now().isoformat(),
                'event_count': len(events),
                'format': 'json'
            },
            'events': events
        }
        
        if compress:
            with gzip.open(export_path, 'wt', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        else:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    async def _export_csv(self, events: List[Dict[str, Any]], export_path: Path, compress: bool):
        """Export events to CSV format"""
        import csv
        
        if not events:
            return
        
        # Define CSV columns
        columns = [
            'event_id', 'event_type', 'timestamp', 'user_id', 'session_id',
            'client_ip', 'action', 'resource', 'result', 'severity', 'risk_score'
        ]
        
        def write_csv(f):
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            for event in events:
                writer.writerow(event)
        
        if compress:
            with gzip.open(export_path, 'wt', encoding='utf-8') as f:
                write_csv(f)
        else:
            with open(export_path, 'w', encoding='utf-8', newline='') as f:
                write_csv(f)
    
    def _setup_file_logger(self):
        """Setup file-based logging for audit system itself"""
        audit_log_file = self.audit_dir / "audit_system.log"
        handler = logging.FileHandler(audit_log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        async def flush_loop():
            while True:
                try:
                    await self._flush_event_buffer()
                    await asyncio.sleep(self.buffer_flush_interval)
                except Exception as e:
                    self.logger.error(f"Buffer flush task error: {e}")
                    await asyncio.sleep(10)
        
        async def cleanup_loop():
            while True:
                try:
                    await self._cleanup_old_logs()
                    await asyncio.sleep(86400)  # Daily cleanup
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
                    await asyncio.sleep(3600)  # Retry in 1 hour
        
        self._flush_task = asyncio.create_task(flush_loop())
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_old_logs(self):
        """Clean up old log files based on retention policy"""
        retention_cutoff = datetime.now() - timedelta(days=self.config.log_retention_days)
        
        deleted_count = 0
        for log_file in self.logs_dir.glob("audit_*.json*"):
            try:
                # Extract timestamp from filename
                file_timestamp_str = log_file.stem.split('_', 1)[1].split('.')[0]
                file_timestamp = datetime.strptime(file_timestamp_str, "%Y%m%d_%H%M%S")
                
                if file_timestamp < retention_cutoff:
                    # Archive before deletion
                    archive_path = self.archive_dir / log_file.name
                    log_file.rename(archive_path)
                    deleted_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Could not process log file {log_file}: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"Archived {deleted_count} old log files")
    
    async def shutdown(self):
        """Shutdown audit logger"""
        try:
            # Cancel background tasks
            if self._flush_task:
                self._flush_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._monitoring_task:
                self._monitoring_task.cancel()
            
            # Final buffer flush
            await self._flush_event_buffer()
            
            # Log shutdown
            await self.log_event(
                event_type=AuditEventType.SYSTEM_SHUTDOWN,
                action="audit_logger_shutdown",
                result="success"
            )
            
            # Final flush
            await self._flush_event_buffer()
            
            self.logger.info("Debug audit logger shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Audit logger shutdown error: {e}")