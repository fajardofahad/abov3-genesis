#!/usr/bin/env python3
"""
Debug ABOV3 Genesis - Find where it's freezing
"""

import sys
import os
from pathlib import Path

print("🔍 Debug: Starting ABOV3 Genesis debug...")

# Set UTF-8 encoding for Windows
if os.name == 'nt':
    print("🔍 Debug: Setting Windows UTF-8 encoding...")
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

def main():
    print("🔍 Debug: In main() function")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    print(f"🔍 Debug: Script directory: {script_dir}")
    
    # Add the project root to Python path
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    print(f"🔍 Debug: Python path updated")
    
    # Check basic imports first
    print("🔍 Debug: Testing basic imports...")
    
    try:
        print("🔍 Debug: Importing asyncio...")
        import asyncio
        print("✅ Debug: asyncio imported successfully")
        
        print("🔍 Debug: Importing rich...")
        from rich.console import Console
        print("✅ Debug: rich imported successfully")
        
        print("🔍 Debug: Importing pathlib...")
        from pathlib import Path
        print("✅ Debug: pathlib imported successfully")
        
        print("🔍 Debug: Testing abov3.main import...")
        from abov3.main import main as abov3_main
        print("✅ Debug: abov3.main imported successfully")
        
        print("🔍 Debug: All imports successful, starting ABOV3...")
        print("🚀 Starting ABOV3 Genesis...")
        
        # Call the main function
        abov3_main()
        
    except ImportError as e:
        print(f"❌ Debug: Import Error: {e}")
        print("\n🔧 Missing dependencies. Please install:")
        print("pip install -r requirements.txt")
        
        # Show which specific import failed
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Debug: Runtime Error: {e}")
        
        # Show full traceback
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("🔍 Debug: Script starting...")
    main()
    print("🔍 Debug: Script completed")