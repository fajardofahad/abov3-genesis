"""
ABOV3 Genesis - Enhanced Ollama Integration Example
Demonstrates how to use the optimized Ollama integration for Claude-level performance
"""

import asyncio
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demonstrate_enhanced_ollama():
    """Demonstrate the enhanced Ollama integration capabilities"""
    
    print("🚀 ABOV3 Genesis Enhanced Ollama Integration Demo")
    print("=" * 60)
    
    try:
        # Initialize the enhanced assistant
        from abov3.core.enhanced_assistant import create_enhanced_assistant
        
        print("🔧 Initializing Enhanced AI Assistant...")
        assistant = await create_enhanced_assistant(Path.cwd())
        print("✅ Assistant initialized successfully!")
        
        # Demonstrate different types of requests
        await demo_code_generation(assistant)
        await demo_debugging(assistant)
        await demo_code_review(assistant)
        await demo_architecture_design(assistant)
        await demo_learning_feedback(assistant)
        await demo_multi_model_comparison(assistant)
        
        # Show system status
        await demo_system_status(assistant)
        
        # Cleanup
        await assistant.cleanup()
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

async def demo_code_generation(assistant):
    """Demonstrate optimized code generation"""
    print("\n💻 Code Generation Demo")
    print("-" * 30)
    
    requests = [
        {
            "prompt": "Create a Python class for managing a TODO list with add, remove, mark complete, and list methods. Include error handling and documentation.",
            "description": "Python Class with CRUD Operations"
        },
        {
            "prompt": "Build a React component for user authentication with email/password login, validation, and error handling.",
            "description": "React Authentication Component"
        },
        {
            "prompt": "Write a FastAPI endpoint for uploading files with size validation, type checking, and secure storage.",
            "description": "FastAPI File Upload Endpoint"
        }
    ]
    
    for i, request in enumerate(requests, 1):
        print(f"\n🔹 Example {i}: {request['description']}")
        
        try:
            response = await assistant.chat(
                message=request["prompt"],
                task_type="code_generation",
                user_id="demo_user"
            )
            
            if response["success"]:
                print(f"   ✅ Generated successfully!")
                print(f"   📊 Quality Score: {response.get('quality_score', 0):.3f}")
                print(f"   ⚡ Processing Time: {response['processing_time_ms']:.0f}ms")
                print(f"   🤖 Model Used: {response.get('model_used', 'unknown')}")
                print(f"   📝 Response Length: {len(response['response'])} chars")
                
                # Show first few lines of response
                lines = response['response'].split('\n')[:5]
                print("   📄 Preview:")
                for line in lines:
                    print(f"      {line}")
                if len(response['response'].split('\n')) > 5:
                    print("      ...")
            else:
                print(f"   ❌ Failed: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        await asyncio.sleep(1)  # Brief pause between requests

async def demo_debugging(assistant):
    """Demonstrate debugging capabilities"""
    print("\n🔍 Debugging Demo")
    print("-" * 20)
    
    bug_examples = [
        {
            "code": """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# This crashes with empty list
result = calculate_average([])
print(result)
""",
            "description": "Division by Zero Bug",
            "prompt": "Fix this Python function that crashes when given an empty list:"
        },
        {
            "code": """
async function fetchUserData(userId) {
    const response = fetch(`/api/users/${userId}`);
    const userData = response.json();
    return userData;
}
""",
            "description": "Missing Await Bug", 
            "prompt": "Debug this JavaScript async function that doesn't work properly:"
        }
    ]
    
    for i, example in enumerate(bug_examples, 1):
        print(f"\n🔹 Debug Example {i}: {example['description']}")
        
        full_prompt = f"{example['prompt']}\n```\n{example['code']}```"
        
        try:
            response = await assistant.chat(
                message=full_prompt,
                task_type="debugging",
                user_id="demo_user"
            )
            
            if response["success"]:
                print(f"   ✅ Debug analysis completed!")
                print(f"   📊 Quality Score: {response.get('quality_score', 0):.3f}")
                print(f"   ⚡ Processing Time: {response['processing_time_ms']:.0f}ms")
                print(f"   🤖 Model Used: {response.get('model_used', 'unknown')}")
                
                # Show brief excerpt
                lines = response['response'].split('\n')[:3]
                print("   🔧 Debug Analysis Preview:")
                for line in lines:
                    if line.strip():
                        print(f"      {line}")
            else:
                print(f"   ❌ Failed: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        await asyncio.sleep(1)

async def demo_code_review(assistant):
    """Demonstrate code review capabilities"""
    print("\n📋 Code Review Demo")
    print("-" * 20)
    
    code_to_review = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

def save_to_file(data, filename):
    f = open(filename, 'w')
    f.write(str(data))
    f.close()

class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def get_info(self):
        return self.name + " - " + self.email
"""
    
    prompt = f"Please review this Python code for best practices, potential issues, and improvements:\n```python\n{code_to_review}\n```"
    
    try:
        response = await assistant.chat(
            message=prompt,
            task_type="code_review",
            user_id="demo_user"
        )
        
        if response["success"]:
            print("   ✅ Code review completed!")
            print(f"   📊 Quality Score: {response.get('quality_score', 0):.3f}")
            print(f"   ⚡ Processing Time: {response['processing_time_ms']:.0f}ms")
            print(f"   🤖 Model Used: {response.get('model_used', 'unknown')}")
            
            # Show key points from review
            lines = response['response'].split('\n')[:6]
            print("   📝 Review Summary:")
            for line in lines:
                if line.strip():
                    print(f"      {line}")
        else:
            print(f"   ❌ Failed: {response.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

async def demo_architecture_design(assistant):
    """Demonstrate architecture design capabilities"""
    print("\n🏗️  Architecture Design Demo")
    print("-" * 30)
    
    architecture_request = """
Design a scalable microservices architecture for a social media platform that needs to handle:
- User management and authentication
- Content creation and sharing (posts, images, videos)
- Real-time messaging and notifications
- Analytics and reporting
- Mobile app support

The system should handle 1 million daily active users and be cloud-native.
"""
    
    try:
        response = await assistant.chat(
            message=architecture_request,
            task_type="architecture",
            user_id="demo_user"
        )
        
        if response["success"]:
            print("   ✅ Architecture design completed!")
            print(f"   📊 Quality Score: {response.get('quality_score', 0):.3f}")
            print(f"   ⚡ Processing Time: {response['processing_time_ms']:.0f}ms")
            print(f"   🤖 Model Used: {response.get('model_used', 'unknown')}")
            print(f"   📄 Response Length: {len(response['response'])} chars")
            
            # Show architecture overview
            lines = response['response'].split('\n')[:8]
            print("   🏛️  Architecture Overview:")
            for line in lines:
                if line.strip():
                    print(f"      {line}")
        else:
            print(f"   ❌ Failed: {response.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

async def demo_learning_feedback(assistant):
    """Demonstrate learning and feedback capabilities"""
    print("\n🧠 Learning System Demo")
    print("-" * 25)
    
    # Generate a response to provide feedback on
    test_prompt = "Create a simple Python function to validate email addresses using regex"
    
    try:
        response = await assistant.chat(
            message=test_prompt,
            task_type="code_generation",
            user_id="demo_user"
        )
        
        if response["success"]:
            session_id = response["session_id"]
            print("   ✅ Generated test response for feedback demo")
            
            # Demonstrate different types of feedback
            feedback_examples = [
                {
                    "type": "thumbs_up",
                    "value": True,
                    "description": "Positive feedback (thumbs up)"
                },
                {
                    "type": "rating", 
                    "value": 4,
                    "description": "4-star rating"
                },
                {
                    "type": "comment",
                    "value": "Good implementation but could use more comprehensive regex pattern",
                    "description": "Detailed comment feedback"
                }
            ]
            
            for feedback in feedback_examples:
                try:
                    success = await assistant.record_feedback(
                        session_id=session_id,
                        message_id="demo_feedback",
                        feedback_type=feedback["type"],
                        feedback_value=feedback["value"],
                        comments="Demo feedback from example script"
                    )
                    
                    status = "✅" if success else "❌"
                    print(f"   {status} {feedback['description']}: {'Recorded' if success else 'Failed'}")
                    
                except Exception as e:
                    print(f"   ❌ Feedback failed: {e}")
            
            # Show learning statistics if available
            if hasattr(assistant, 'learning_system') and assistant.learning_system:
                try:
                    report = assistant.learning_system.get_learning_report()
                    print(f"   📊 Learning Stats: {report['overview']['total_feedback_entries']} feedback entries")
                except Exception as e:
                    print(f"   ⚠️  Could not get learning stats: {e}")
                    
        else:
            print(f"   ❌ Could not generate test response: {response.get('error')}")
            
    except Exception as e:
        print(f"   ❌ Learning demo error: {e}")

async def demo_multi_model_comparison(assistant):
    """Demonstrate multi-model capabilities"""
    print("\n🔄 Multi-Model Comparison Demo")
    print("-" * 35)
    
    # Get system status to see available models
    try:
        system_status = assistant.get_system_status()
        
        if "model_manager_status" in system_status:
            mm_status = system_status["model_manager_status"]
            available_models = [model for model, status in mm_status.get("model_statuses", {}).items() 
                              if status == "available"]
            
            print(f"   📋 Available Models: {', '.join(available_models) if available_models else 'None detected'}")
            
            if available_models:
                test_prompt = "Create a Python function to merge two sorted arrays"
                
                # Test with different models (if multiple available)
                for model in available_models[:2]:  # Test first 2 models
                    try:
                        response = await assistant.chat(
                            message=test_prompt,
                            task_type="code_generation",
                            preferences={"preferred_model": model},
                            user_id="model_comparison_user"
                        )
                        
                        if response["success"]:
                            print(f"   🤖 {model}:")
                            print(f"      Quality: {response.get('quality_score', 0):.3f}")
                            print(f"      Time: {response['processing_time_ms']:.0f}ms")
                            print(f"      Confidence: {response.get('model_confidence', 0):.3f}")
                        else:
                            print(f"   ❌ {model}: Failed - {response.get('error')}")
                            
                    except Exception as e:
                        print(f"   ❌ {model}: Error - {e}")
                    
                    await asyncio.sleep(1)
            else:
                print("   ⚠️  No models available for comparison")
        else:
            print("   ⚠️  Model manager not available")
            
    except Exception as e:
        print(f"   ❌ Multi-model demo error: {e}")

async def demo_system_status(assistant):
    """Demonstrate system status and monitoring"""
    print("\n📊 System Status Demo")
    print("-" * 25)
    
    try:
        status = assistant.get_system_status()
        
        print("   🔧 Core System:")
        print(f"      Initialized: {status.get('initialized', False)}")
        print(f"      Active Sessions: {status.get('active_sessions', 0)}")
        
        # Component status
        components = status.get('components', {})
        loaded_components = sum(1 for v in components.values() if v)
        total_components = len(components)
        print(f"      Components: {loaded_components}/{total_components} loaded")
        
        for component, loaded in components.items():
            status_icon = "✅" if loaded else "❌"
            print(f"        {status_icon} {component.replace('_', ' ').title()}")
        
        # Performance stats
        perf_stats = status.get('performance_stats', {})
        if perf_stats:
            print("   ⚡ Performance:")
            print(f"      Total Requests: {perf_stats.get('total_requests', 0)}")
            print(f"      Success Rate: {perf_stats.get('successful_requests', 0)}/{perf_stats.get('total_requests', 0)}")
            print(f"      Average Quality: {perf_stats.get('average_quality', 0):.3f}")
            print(f"      Average Response Time: {perf_stats.get('average_response_time', 0):.2f}s")
        
        # Model manager status
        if 'model_manager_status' in status:
            mm_status = status['model_manager_status']
            print("   🤖 Model Manager:")
            print(f"      Models Available: {mm_status.get('models_available', 0)}")
            print(f"      System Success Rate: {mm_status.get('success_rate', 0):.3f}")
            print(f"      Average Response Time: {mm_status.get('avg_selection_time_ms', 0):.1f}ms")
        
        # Learning system status
        if 'learning_stats' in status:
            learning_stats = status['learning_stats']
            overview = learning_stats.get('overview', {})
            print("   🧠 Learning System:")
            print(f"      Feedback Entries: {overview.get('total_feedback_entries', 0)}")
            print(f"      Models Tracked: {overview.get('models_tracked', 0)}")
            print(f"      Patterns Learned: {overview.get('patterns_learned', 0)}")
        
    except Exception as e:
        print(f"   ❌ Status demo error: {e}")

async def interactive_demo():
    """Interactive demo allowing user input"""
    print("\n🎮 Interactive Demo Mode")
    print("-" * 25)
    print("Type your requests below. Commands: 'quit' to exit, 'status' for system info")
    
    try:
        from abov3.core.enhanced_assistant import create_enhanced_assistant
        assistant = await create_enhanced_assistant(Path.cwd())
        
        session_id = None
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'status':
                    status = assistant.get_system_status()
                    print(f"📊 System Status: {json.dumps(status, indent=2, default=str)}")
                    continue
                
                if not user_input:
                    continue
                
                print("🤔 Thinking...")
                response = await assistant.chat(
                    message=user_input,
                    session_id=session_id,
                    user_id="interactive_user"
                )
                
                if response["success"]:
                    print(f"\n🤖 Assistant: {response['response']}")
                    print(f"\n📊 Stats: Quality: {response.get('quality_score', 0):.3f} | "
                          f"Time: {response['processing_time_ms']:.0f}ms | "
                          f"Model: {response.get('model_used', 'unknown')}")
                    
                    session_id = response["session_id"]
                else:
                    print(f"\n❌ Error: {response['error']}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        await assistant.cleanup()
        
    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")

async def main():
    """Main demo function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_demo()
    else:
        await demonstrate_enhanced_ollama()

if __name__ == "__main__":
    asyncio.run(main())