#!/usr/bin/env python3
"""
ABOV3 Genesis - Direct Runner
Run ABOV3 Genesis without installation
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Add the project root to Python path
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    # Check if required modules can be imported
    try:
        from abov3.main import main as abov3_main
        print("🚀 Starting ABOV3 Genesis...")
        abov3_main()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 Please install dependencies first:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()