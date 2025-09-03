#!/usr/bin/env python3
"""
Simulate the exact user flow to reproduce the division by zero error
"""
import subprocess
import time
import sys

print("Simulating user flow: python run_abov3.py")
print("Will enter: 1, then n, then 'make me a python hello world code'")
print("=" * 60)

# Create input for the application
user_input = "1\nn\nmake me a python hello world code\nexit\n"

# Run the application with the input
try:
    process = subprocess.Popen(
        [sys.executable, "run_abov3.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=r"C:\Users\fajar\Documents\ABOV3\abov3-Genesis\abov3-genesis-v1.0.0"
    )
    
    # Send input and get output
    stdout, stderr = process.communicate(input=user_input, timeout=10)
    
    print("STDOUT:")
    print(stdout)
    print("\nSTDERR:")
    print(stderr)
    
    # Look for division by zero errors
    if "division by zero" in stdout.lower() or "division by zero" in stderr.lower():
        print("\n[FOUND] Division by zero error detected!")
        
except subprocess.TimeoutExpired:
    print("Process timed out - this might be normal for interactive apps")
    process.kill()
    stdout, stderr = process.communicate()
    print("STDOUT (partial):")
    print(stdout if stdout else "No output")
    print("\nSTDERR (partial):")
    print(stderr if stderr else "No errors")
    
except Exception as e:
    print(f"Error running application: {e}")