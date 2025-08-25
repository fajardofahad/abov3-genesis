#!/usr/bin/env python3
"""Debug pattern matching issues"""

import re
import sys
import io

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Test text
text = "write a python hello world script"
print(f"Testing text: '{text}'")

# Test patterns from _detect_code_generation
code_patterns = [
    r'(?:write|create|generate|make)\s+(?:me\s+)?(?:a|an|some)?\s*(?:code|script|function|class|program)',
    r'(?:write|create|generate)\s+.*\.(py|js|html|css|java|cpp|c|rs|go|php|rb|ts|jsx|tsx)',
    r'(?:python|javascript|java|cpp|rust|go|ruby|php)\s+(?:code|script|program|function)',
    r'implement\s+(?:a|an|the)?\s*(?:function|class|method|algorithm)',
    r'code\s+(?:for|to)\s+(?:do|perform|handle|process)',
    r'hello\s+world\s+(?:in|using|with)',
    r'example\s+(?:code|script|program)'
]

print("\nTesting code patterns:")
for i, pattern in enumerate(code_patterns, 1):
    match = re.search(pattern, text)
    if match:
        print(f"  Pattern {i}: MATCH - {pattern}")
        print(f"    Matched: '{match.group()}'")
    else:
        print(f"  Pattern {i}: NO MATCH - {pattern}")

# Test the first pattern specifically
print("\n\nDetailed test of first pattern:")
pattern = r'(?:write|create|generate|make)\s+(?:me\s+)?(?:a|an|some)?\s*(?:code|script|function|class|program)'
print(f"Pattern: {pattern}")
print(f"Text: '{text}'")

# Try simpler patterns
print("\nTrying simpler patterns:")
simple_patterns = [
    r'write',
    r'script',
    r'write.*script',
    r'write\s+a',
    r'python\s+hello',
    r'hello\s+world'
]

for pattern in simple_patterns:
    match = re.search(pattern, text)
    print(f"  '{pattern}': {'MATCH' if match else 'NO MATCH'}")

# Check if it's a string encoding issue
print("\nString details:")
print(f"  Type: {type(text)}")
print(f"  Encoding: {sys.getdefaultencoding()}")
print(f"  Bytes: {text.encode('utf-8')}")