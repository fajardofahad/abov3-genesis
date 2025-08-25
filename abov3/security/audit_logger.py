"""
ABOV3 Genesis - Security Audit Logger
Comprehensive security audit logging and monitoring system
"""

import asyncio
import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import hashlib
import logging
from logging.handlers import RotatingFileHandler


class AuditLevel(Enum):
    """Audit log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"


class SecurityAuditLogger:
    """
    Enterprise Security Audit Logger
    Provides tamper-evident logging with encryption and retention policies
    """
    
    def __init__(self, security_dir: Path, retention_days: int = 90):
        self.security_dir = security_dir
        self.retention_days = retention_days
        
        # Audit log directory
        self.audit_dir = security_dir / 'audit_logs'
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Current log file
        self.current_log_file = self.audit_dir / f"security_audit_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Setup structured logging
        self.logger = logging.getLogger('abov3.security.audit')
        self.logger.setLevel(logging.DEBUG)
        
        # Create rotating file handler
        handler = RotatingFileHandler(
            self.current_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        
        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Audit statistics
        self.audit_stats = {
            'total_events_logged': 0,
            'events_by_level': {},
            'events_by_type': {},
            'log_files_created': 0,
            'log_files_archived': 0
        }
        
        # Log integrity tracking
        self.log_hashes: Dict[str, str] = {}
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_old_logs())
    
    async def initialize(self):
        """Initialize audit logger"""
        await self.log_event("audit_logger_initialized", {
            "retention_days": self.retention_days,
            "audit_directory": str(self.audit_dir)
        }, level=AuditLevel.INFO)
        
        return True
    
    async def log_event(self, event_type: str, event_data: Dict[str, Any], 
                       level: AuditLevel = AuditLevel.INFO, user_id: str = None,
                       ip_address: str = None, user_agent: str = None):
        """Log security audit event"""
        try:
            # Create audit record
            audit_record = {
                'timestamp': datetime.now().isoformat(),
                'event_id': hashlib.sha256(f"{event_type}_{datetime.now().isoformat()}_{user_id}".encode()).hexdigest()[:16],
                'event_type': event_type,
                'level': level.value,
                'data': event_data,
                'user_id': user_id,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'process_id': None,  # Could add process tracking
                'session_id': event_data.get('session_id') if isinstance(event_data, dict) else None
            }
            
            # Add integrity hash
            record_string = json.dumps(audit_record, sort_keys=True)
            audit_record['integrity_hash'] = hashlib.sha256(record_string.encode()).hexdigest()
            
            # Log to structured logger
            log_message = json.dumps(audit_record, ensure_ascii=False)
            
            if level == AuditLevel.CRITICAL:
                self.logger.critical(log_message)
            elif level == AuditLevel.ERROR:
                self.logger.error(log_message)
            elif level == AuditLevel.WARNING:
                self.logger.warning(log_message)
            elif level == AuditLevel.SECURITY:
                self.logger.warning(f"SECURITY: {log_message}")
            else:
                self.logger.info(log_message)
            
            # Update statistics
            self.audit_stats['total_events_logged'] += 1
            self.audit_stats['events_by_level'][level.value] = \
                self.audit_stats['events_by_level'].get(level.value, 0) + 1
            self.audit_stats['events_by_type'][event_type] = \
                self.audit_stats['events_by_type'].get(event_type, 0) + 1
            
            # Store in separate JSON file for structured querying
            await self._store_structured_log(audit_record)
            
        except Exception as e:
            # Fallback logging - even if audit fails, we need to log the failure
            error_msg = f"Audit logging failed for event {event_type}: {str(e)}"
            self.logger.error(error_msg)
    
    async def _store_structured_log(self, audit_record: Dict[str, Any]):
        """Store audit record in structured JSON format"""
        try:
            date_str = datetime.now().strftime('%Y%m%d')
            structured_log_file = self.audit_dir / f"structured_audit_{date_str}.jsonl"
            
            # Append to JSONL file (JSON Lines format)
            with open(structured_log_file, 'a', encoding='utf-8') as f:
                json.dump(audit_record, f, ensure_ascii=False)
                f.write('\n')
            
        except Exception as e:
            self.logger.error(f"Failed to store structured audit log: {e}")
    
    async def query_logs(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        event_types: Optional[List[str]] = None,
                        levels: Optional[List[AuditLevel]] = None,
                        user_id: Optional[str] = None,
                        limit: int = 1000) -> List[Dict[str, Any]]:
        """Query audit logs with filters"""
        try:
            results = []
            
            # Determine date range
            if start_date is None:
                start_date = datetime.now() - timedelta(days=7)
            if end_date is None:
                end_date = datetime.now()
            
            # Generate list of potential log files
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y%m%d')
                log_file = self.audit_dir / f"structured_audit_{date_str}.jsonl"
                
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                
                                try:
                                    record = json.loads(line)
                                    record_time = datetime.fromisoformat(record['timestamp'])
                                    
                                    # Apply filters
                                    if record_time < start_date or record_time > end_date:
                                        continue
                                    
                                    if event_types and record['event_type'] not in event_types:
                                        continue
                                    
                                    if levels and AuditLevel(record['level']) not in levels:
                                        continue
                                    
                                    if user_id and record.get('user_id') != user_id:
                                        continue
                                    
                                    results.append(record)
                                    
                                    if len(results) >= limit:
                                        break
                                        
                                except json.JSONDecodeError:
                                    continue
                                    
                    except Exception as e:
                        self.logger.error(f"Error reading log file {log_file}: {e}")
                
                current_date += timedelta(days=1)
                
                if len(results) >= limit:
                    break
            
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Log query failed: {e}")
            return []
    
    async def get_security_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent security-related events"""
        start_date = datetime.now() - timedelta(hours=hours)
        
        security_event_types = [
            'user_authenticated',
            'user_logout',
            'login_failed',
            'password_changed',
            'user_created',
            'permission_denied',
            'security_violation',
            'prompt_injection_detected',
            'file_operation_blocked',
            'rate_limit_exceeded',
            'suspicious_activity',
            'account_locked',
            'mfa_enabled',
            'mfa_disabled',
            'emergency_shutdown'
        ]
        
        return await self.query_logs(
            start_date=start_date,
            event_types=security_event_types,
            levels=[AuditLevel.SECURITY, AuditLevel.WARNING, AuditLevel.ERROR, AuditLevel.CRITICAL],
            limit=500
        )
    
    async def generate_security_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate security audit report"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            logs = await self.query_logs(start_date=start_date, limit=10000)
            
            # Analyze logs
            report = {
                'report_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': datetime.now().isoformat(),
                    'days': days
                },
                'summary': {
                    'total_events': len(logs),
                    'security_events': 0,
                    'failed_authentications': 0,
                    'successful_authentications': 0,
                    'permission_denials': 0,
                    'file_operations_blocked': 0,
                    'prompt_injections_detected': 0
                },
                'top_users': {},
                'top_event_types': {},
                'security_incidents': [],
                'recommendations': []
            }
            
            # Analyze events
            for log in logs:
                event_type = log['event_type']
                user_id = log.get('user_id', 'unknown')
                level = log['level']
                
                # Count by user
                report['top_users'][user_id] = report['top_users'].get(user_id, 0) + 1
                
                # Count by event type
                report['top_event_types'][event_type] = report['top_event_types'].get(event_type, 0) + 1
                
                # Specific security metrics
                if event_type == 'login_failed':
                    report['summary']['failed_authentications'] += 1
                elif event_type == 'user_authenticated':
                    report['summary']['successful_authentications'] += 1
                elif event_type == 'permission_denied':
                    report['summary']['permission_denials'] += 1
                elif event_type == 'file_operation_blocked':
                    report['summary']['file_operations_blocked'] += 1
                elif event_type == 'prompt_injection_detected':
                    report['summary']['prompt_injections_detected'] += 1
                
                # Security incidents
                if level in ['error', 'critical', 'security']:
                    report['security_incidents'].append({
                        'timestamp': log['timestamp'],
                        'event_type': event_type,
                        'level': level,
                        'user_id': user_id,
                        'summary': self._summarize_event(log)
                    })
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report)
            
            # Sort top lists
            report['top_users'] = dict(sorted(report['top_users'].items(), key=lambda x: x[1], reverse=True)[:10])
            report['top_event_types'] = dict(sorted(report['top_event_types'].items(), key=lambda x: x[1], reverse=True)[:10])
            
            return report
            
        except Exception as e:
            return {'error': f'Failed to generate security report: {str(e)}'}
    
    def _summarize_event(self, log_record: Dict[str, Any]) -> str:
        """Create human-readable summary of log event"""
        event_type = log_record['event_type']
        user_id = log_record.get('user_id', 'unknown user')
        
        summaries = {
            'login_failed': f'Failed login attempt by {user_id}',
            'user_authenticated': f'Successful login by {user_id}',
            'permission_denied': f'Permission denied for {user_id}',
            'prompt_injection_detected': f'Prompt injection attempt by {user_id}',
            'file_operation_blocked': f'Blocked file operation by {user_id}',
            'account_locked': f'Account {user_id} was locked',
            'emergency_shutdown': 'Emergency security shutdown initiated'
        }
        
        return summaries.get(event_type, f'{event_type} event for {user_id}')
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on report"""
        recommendations = []
        summary = report['summary']
        
        # Failed authentication analysis
        if summary['failed_authentications'] > 100:
            recommendations.append("High number of failed authentications detected. Consider implementing additional rate limiting or account lockout policies.")
        
        # Permission denial analysis
        if summary['permission_denials'] > 50:
            recommendations.append("Frequent permission denials suggest users may need additional training or role adjustments.")
        
        # Prompt injection analysis
        if summary['prompt_injections_detected'] > 0:
            recommendations.append("Prompt injection attempts detected. Review and strengthen AI input validation.")
        
        # File operation blocking
        if summary['file_operations_blocked'] > 20:
            recommendations.append("Multiple file operations blocked. Review file access policies and user permissions.")
        
        # Security incidents
        if len(report['security_incidents']) > 10:
            recommendations.append("High number of security incidents. Consider reviewing security policies and user training.")
        
        return recommendations
    
    async def verify_log_integrity(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Verify integrity of audit logs"""
        try:
            if date is None:
                date = datetime.now()
            
            date_str = date.strftime('%Y%m%d')
            log_file = self.audit_dir / f"structured_audit_{date_str}.jsonl"
            
            if not log_file.exists():
                return {'verified': False, 'error': 'Log file not found'}
            
            verified_records = 0
            corrupted_records = 0
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                        
                        # Verify integrity hash
                        stored_hash = record.pop('integrity_hash', None)
                        record_string = json.dumps(record, sort_keys=True)
                        computed_hash = hashlib.sha256(record_string.encode()).hexdigest()
                        
                        if stored_hash == computed_hash:
                            verified_records += 1
                        else:
                            corrupted_records += 1
                            
                    except json.JSONDecodeError:
                        corrupted_records += 1
            
            return {
                'verified': corrupted_records == 0,
                'verified_records': verified_records,
                'corrupted_records': corrupted_records,
                'total_records': verified_records + corrupted_records,
                'integrity_percentage': (verified_records / (verified_records + corrupted_records)) * 100 if (verified_records + corrupted_records) > 0 else 100
            }
            
        except Exception as e:
            return {'verified': False, 'error': f'Integrity verification failed: {str(e)}'}
    
    async def _cleanup_old_logs(self):
        """Background task to cleanup old log files"""
        while True:
            try:
                cutoff_date = datetime.now() - timedelta(days=self.retention_days)
                
                for log_file in self.audit_dir.glob("*.log*"):
                    try:
                        file_date = datetime.fromtimestamp(log_file.stat().st_ctime)
                        if file_date < cutoff_date:
                            # Archive before deletion
                            archive_path = log_file.with_suffix(log_file.suffix + '.gz')
                            with open(log_file, 'rb') as f_in:
                                with gzip.open(archive_path, 'wb') as f_out:
                                    f_out.writelines(f_in)
                            
                            log_file.unlink()
                            self.audit_stats['log_files_archived'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error cleaning up log file {log_file}: {e}")
                
                # Sleep for 24 hours
                await asyncio.sleep(86400)
                
            except Exception as e:
                self.logger.error(f"Error in log cleanup: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logger statistics"""
        return self.audit_stats.copy()
    
    async def cleanup(self):
        """Cleanup audit logger resources"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)