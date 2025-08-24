#!/usr/bin/env python3
"""
ABOV3 Genesis - Setup Test
Test if all components can be imported and basic functionality works
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

def test_imports():
    """Test if all main modules can be imported"""
    print("🧪 Testing ABOV3 Genesis imports...")
    
    try:
        # Add project to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Test core imports
        from abov3.ui.genz import GenZStatus
        print("  ✅ GenZ Status messages")
        
        from abov3.genesis.engine import GenesisEngine
        print("  ✅ Genesis Engine")
        
        from abov3.project.registry import ProjectRegistry
        print("  ✅ Project Registry")
        
        from abov3.core.ollama_client import OllamaClient
        print("  ✅ Ollama Client")
        
        from abov3.agents.manager import AgentManager
        print("  ✅ Agent Manager")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_genz_messages():
    """Test GenZ status messages"""
    print("\n🎭 Testing GenZ status messages...")
    
    try:
        from abov3.ui.genz import GenZStatus
        
        genz = GenZStatus()
        
        # Test different categories
        thinking = genz.get_thinking_status()
        building = genz.get_building_status()
        success = genz.get_success_status()
        
        print(f"  💭 Thinking: {thinking}")
        print(f"  🏗️  Building: {building}")
        print(f"  ✨ Success: {success}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GenZ test failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection (if available)"""
    print("\n🤖 Testing Ollama connection...")
    
    try:
        from abov3.core.ollama_client import OllamaClient
        import asyncio
        
        async def check_ollama():
            client = OllamaClient()
            is_available = await client.is_available()
            
            if is_available:
                models = await client.list_models()
                print(f"  ✅ Ollama is running with {len(models)} models")
                for model in models[:3]:  # Show first 3 models
                    print(f"    - {model.get('name', 'Unknown')}")
            else:
                print("  ⚠️  Ollama is not running (this is okay for testing)")
            
            return is_available
        
        return asyncio.run(check_ollama())
        
    except Exception as e:
        print(f"  ❌ Ollama test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 ABOV3 Genesis - Setup Test\n")
    
    results = []
    
    # Test imports
    results.append(test_imports())
    
    # Test GenZ messages
    results.append(test_genz_messages())
    
    # Test Ollama (optional)
    results.append(test_ollama_connection())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ABOV3 Genesis is ready to run.")
        print("\n🚀 You can now run:")
        print("   python run_abov3.py")
        print("   # or")
        print("   run_abov3.bat")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n🔧 Try installing dependencies:")
        print("   pip install -r requirements-dev.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)