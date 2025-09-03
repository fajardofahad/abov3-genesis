#!/usr/bin/env python3
"""
Trace the exact location of the persistent division by zero error
"""

import sys
import traceback
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(message)s')

print("TRACING PERSISTENT DIVISION BY ZERO ERROR")
print("=" * 60)

# Monkey-patch division to catch where it happens
original_truediv = float.__truediv__
original_floordiv = float.__floordiv__
original_div = int.__truediv__
original_intdiv = int.__floordiv__

def traced_truediv(self, other):
    if other == 0:
        print(f"\n[CAUGHT] Division by zero: {self} / {other}")
        traceback.print_stack()
        print()
    return original_truediv(self, other)

def traced_floordiv(self, other):
    if other == 0:
        print(f"\n[CAUGHT] Floor division by zero: {self} // {other}")
        traceback.print_stack()
        print()
    return original_floordiv(self, other)

def traced_intdiv(self, other):
    if other == 0:
        print(f"\n[CAUGHT] Int division by zero: {self} / {other}")
        traceback.print_stack()
        print()
    return original_div(self, other)

def traced_intfloordiv(self, other):
    if other == 0:
        print(f"\n[CAUGHT] Int floor division by zero: {self} // {other}")
        traceback.print_stack()
        print()
    return original_intdiv(self, other)

# Apply the patches
float.__truediv__ = traced_truediv
float.__floordiv__ = traced_floordiv
int.__truediv__ = traced_intdiv
int.__floordiv__ = traced_intfloordiv

try:
    # Now run the actual code that's causing issues
    from abov3.core.enhanced_debug_integration import EnhancedDebugIntegration
    
    print("Initializing debug integration...")
    debug = EnhancedDebugIntegration('.', mode='development')
    
    print("Simulating error handling...")
    try:
        # Trigger an actual error
        x = 1 / 0
    except ZeroDivisionError as e:
        print(f"Handling error: {e}")
        result = debug.handle_error_sync(e, {'context': 'test'})
        print(f"Result: {result.get('success', False)}")
        
except Exception as e:
    print(f"\n[ERROR CAUGHT]: {e}")
    traceback.print_exc()
    
print("\n" + "=" * 60)
print("Trace complete. Check above for division by zero locations.")