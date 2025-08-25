#!/usr/bin/env python3
"""
ABOV3 Genesis - Production Monitoring & Observability Setup
Enterprise-grade monitoring, alerting, and performance tracking setup
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

from abov3.infrastructure.monitoring import (
    ObservabilityManager,
    AlertManager,
    Alert,
    AlertSeverity,
    MetricsCollector,
    StructuredLogger,
    LogLevel
)

async def setup_production_monitoring(project_path: Path):
    """Setup production monitoring and test it"""
    print("Setting up ABOV3 Genesis Production Monitoring...")
    
    try:
        # Initialize observability manager
        observability_manager = ObservabilityManager(project_path)
        
        print("+ Observability manager initialized")
        
        # Test basic functionality
        await observability_manager.record_application_event(
            "monitoring_setup_test",
            component="monitoring",
            operation="setup"
        )
        
        # Test metrics
        await observability_manager.metrics_collector.increment("app.setup_test")
        await observability_manager.metrics_collector.gauge("app.test_gauge", 42.0)
        
        print("+ Basic metrics recording working")
        
        # Setup some basic alerts
        basic_alerts = [
            Alert(
                alert_id="high_cpu",
                name="High CPU Usage",
                condition="system.cpu.usage_percent > 80",
                severity=AlertSeverity.HIGH,
                threshold=80.0,
                comparison=">",
                metric_name="system.cpu.usage_percent",
                description="System CPU usage is high"
            ),
            Alert(
                alert_id="high_memory",
                name="High Memory Usage",
                condition="system.memory.usage_percent > 90",
                severity=AlertSeverity.CRITICAL,
                threshold=90.0,
                comparison=">",
                metric_name="system.memory.usage_percent",
                description="System memory usage is critical"
            )
        ]
        
        for alert in basic_alerts:
            observability_manager.alert_manager.add_alert(alert)
        
        print(f"+ Configured {len(basic_alerts)} alerts")
        
        # Test dashboard
        dashboard = await observability_manager.get_observability_dashboard()
        
        if dashboard and "system_info" in dashboard:
            print("+ Monitoring dashboard working")
        else:
            print("- Monitoring dashboard incomplete")
        
        # Cleanup
        await observability_manager.cleanup()
        
        return {
            "status": "success",
            "components_configured": 4,
            "alerts_configured": len(basic_alerts),
            "dashboard_working": "system_info" in dashboard
        }
        
    except Exception as e:
        print(f"- Setup failed: {e}")
        return {"status": "error", "error": str(e)}

async def main():
    """Main setup function"""
    print("ABOV3 Genesis - Production Monitoring Setup")
    print("=" * 60)
    
    project_path = Path.cwd()
    
    try:
        result = await setup_production_monitoring(project_path)
        
        if result["status"] == "success":
            print("\n+ Production monitoring setup completed successfully!")
            print(f"Components configured: {result['components_configured']}")
            print(f"Alerts configured: {result['alerts_configured']}")
            print(f"Dashboard working: {result['dashboard_working']}")
        else:
            print(f"- Setup failed: {result['error']}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"- Setup failed with exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))