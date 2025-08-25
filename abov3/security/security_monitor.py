"""
ABOV3 Genesis - Security Monitor
Real-time security monitoring and alerting system
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class SecurityMonitor:
    """Real-time security monitoring system"""
    
    def __init__(self, security_core, audit_logger=None):
        self.security_core = security_core
        self.audit_logger = audit_logger
        self.is_monitoring = False
        self.monitoring_task = None
        
        # Monitoring statistics
        self.monitor_stats = {
            'monitoring_duration': 0,
            'alerts_generated': 0,
            'security_events_detected': 0,
            'system_health_checks': 0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'failed_logins_per_minute': 10,
            'blocked_requests_per_minute': 50,
            'security_violations_per_hour': 20
        }
    
    async def start(self):
        """Start security monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            if self.audit_logger:
                await self.audit_logger.log_event("security_monitoring_started", {
                    "timestamp": datetime.now().isoformat()
                })
    
    async def stop(self):
        """Stop security monitoring"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.audit_logger:
                await self.audit_logger.log_event("security_monitoring_stopped", {
                    "timestamp": datetime.now().isoformat(),
                    "monitoring_duration": self.monitor_stats['monitoring_duration']
                })
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        start_time = datetime.now()
        
        try:
            while self.is_monitoring:
                # Perform security health check
                await self._perform_health_check()
                
                # Check for security alerts
                await self._check_security_alerts()
                
                # Update monitoring duration
                self.monitor_stats['monitoring_duration'] = (datetime.now() - start_time).total_seconds()
                self.monitor_stats['system_health_checks'] += 1
                
                # Sleep for monitoring interval (30 seconds)
                await asyncio.sleep(30)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.audit_logger:
                await self.audit_logger.log_event("security_monitoring_error", {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
    async def _perform_health_check(self):
        """Perform security system health check"""
        # Check security core status
        status = await self.security_core.get_security_status()
        
        # Generate alerts based on status
        if status['overall_status'] == 'critical':
            await self._generate_alert("CRITICAL_SECURITY_STATUS", {
                "message": "Security system in critical state",
                "components": status['components']
            })
    
    async def _check_security_alerts(self):
        """Check for security alert conditions"""
        # This would analyze recent security events and generate alerts
        # Implementation depends on specific alert logic
        pass
    
    async def _generate_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Generate security alert"""
        self.monitor_stats['alerts_generated'] += 1
        
        if self.audit_logger:
            await self.audit_logger.log_event("security_alert", {
                "alert_type": alert_type,
                "alert_data": alert_data,
                "timestamp": datetime.now().isoformat()
            }, level="CRITICAL")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        return {
            'is_monitoring': self.is_monitoring,
            'statistics': self.monitor_stats.copy(),
            'alert_thresholds': self.alert_thresholds.copy()
        }