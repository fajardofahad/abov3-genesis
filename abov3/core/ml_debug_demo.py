"""
ABOV3 Genesis - ML Debug Engine Demonstration
Showcase of Claude-level intelligent debugging capabilities
"""

import sys
import json
import traceback
from datetime import datetime
from typing import Any, Dict, List

# Import enhanced debugger
try:
    from .enhanced_ml_debugger import get_enhanced_debugger, debug_with_ml, ask_debug_question, generate_tests
    from .enterprise_debugger import get_enhanced_debug_engine
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("Enhanced ML debugger not available - using basic debugging")


def demonstrate_ml_error_analysis():
    """Demonstrate ML-powered error analysis"""
    print("\n" + "="*80)
    print("🧠 ML-POWERED ERROR ANALYSIS DEMONSTRATION")
    print("="*80)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Enhanced ML debugger not available")
        return
    
    # Sample problematic code
    problematic_code = '''
def process_data(data):
    """Process user data and return formatted result"""
    result = {}
    
    # This will cause AttributeError if data is None
    for item in data.items():
        key = item[0]
        value = item[1]
        
        # This will cause KeyError if 'name' doesn't exist
        result[key] = value['name'].upper()
    
    return result

# This will cause an error
user_data = None
processed = process_data(user_data)
'''
    
    print("📝 Analyzing problematic code:")
    print(problematic_code[:200] + "...")
    
    try:
        # Create debug session
        debugger = get_enhanced_debugger()
        session_id = debugger.create_debug_session(problematic_code, "demo.py")
        
        # Simulate the error
        exec(problematic_code)
        
    except Exception as e:
        print(f"\n🔍 Error caught: {type(e).__name__}: {e}")
        
        # Analyze with ML
        analysis = debugger.analyze_error_with_ml(e, problematic_code)
        
        print("\n🤖 ML Analysis Results:")
        print(f"   Confidence: {analysis.get('confidence', 0):.1%}")
        print(f"   Error Type: {analysis.get('error_type', 'Unknown')}")
        
        # Show fix suggestions
        fix_suggestions = analysis.get('fix_suggestions', [])
        if fix_suggestions:
            print("\n💡 Intelligent Fix Suggestions:")
            for i, fix in enumerate(fix_suggestions[:3], 1):
                print(f"\n   {i}. {fix.get('explanation', 'No explanation')}")
                print(f"      Confidence: {fix.get('confidence', 0):.1%}")
                print(f"      Code Fix:\n{fix.get('code', '# No code provided')}")
        
        # Show ML insights
        ml_analysis = analysis.get('ml_analysis', {})
        if ml_analysis:
            print("\n🔬 Advanced ML Insights:")
            patterns = ml_analysis.get('error_patterns', [])
            for pattern in patterns[:2]:
                print(f"   • {pattern.get('description', 'Unknown pattern')}")
            
            similar_errors = ml_analysis.get('similar_errors', [])
            if similar_errors:
                print(f"   • Found {len(similar_errors)} similar error patterns")


def demonstrate_natural_language_interface():
    """Demonstrate natural language debugging interface"""
    print("\n" + "="*80)
    print("💬 NATURAL LANGUAGE DEBUGGING DEMONSTRATION")
    print("="*80)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Natural language interface not available")
        return
    
    # Sample code with issues
    sample_code = '''
import requests
import json

def fetch_user_data(user_id):
    url = f"https://api.example.com/users/{user_id}"
    response = requests.get(url)
    data = response.json()
    return data['user']['profile']['name']

def process_users(user_ids):
    results = []
    for uid in user_ids:
        user_data = fetch_user_data(uid)
        results.append(user_data)
    return results
'''
    
    print("📝 Sample code loaded for analysis")
    
    # Create debug session
    debugger = get_enhanced_debugger()
    session_id = debugger.create_debug_session(sample_code, "user_service.py")
    
    # Natural language queries to demonstrate
    queries = [
        "What could go wrong with this code?",
        "How can I make this code more robust?",
        "What are the potential security issues?",
        "How can I optimize the performance?",
        "What tests should I write for this code?"
    ]
    
    print("\n🗣️  Natural Language Debugging Session:")
    
    for i, query in enumerate(queries, 1):
        print(f"\n❓ Query {i}: '{query}'")
        
        try:
            response = debugger.ask_natural_language(query)
            
            if 'error' in response:
                print(f"   ❌ Error: {response['error']}")
                continue
            
            print(f"   🤖 Response: {response.get('response', 'No response')[:200]}...")
            print(f"   🎯 Intent: {response.get('intent', 'unknown')}")
            print(f"   📊 Confidence: {response.get('confidence', 0):.1%}")
            
            recommendations = response.get('recommendations', [])
            if recommendations:
                print(f"   💡 Recommendations: {recommendations[0]}")
            
            follow_ups = response.get('follow_up_questions', [])
            if follow_ups:
                print(f"   ❔ Follow-up: {follow_ups[0]}")
        
        except Exception as e:
            print(f"   ❌ Query failed: {e}")


def demonstrate_predictive_debugging():
    """Demonstrate predictive debugging capabilities"""
    print("\n" + "="*80)
    print("🔮 PREDICTIVE DEBUGGING DEMONSTRATION")
    print("="*80)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Predictive debugging not available")
        return
    
    # Code with various quality issues
    risky_code = '''
def complex_function(data, options):
    if data:
        if options:
            if "mode" in options:
                if options["mode"] == "advanced":
                    if "settings" in data:
                        for key in data["settings"]:
                            if key == "critical":
                                value = data["settings"][key]
                                if isinstance(value, dict):
                                    for subkey in value:
                                        if subkey == "level":
                                            level = value[subkey]
                                            if level > 5:
                                                return eval(data["settings"]["formula"])
    return None

def process_input(user_input):
    result = eval(user_input)  # Security risk!
    return result

# Global variables (bad practice)
global_counter = 0
temp_data = {}

def update_global():
    global global_counter, temp_data
    global_counter += 1
    temp_data[str(global_counter)] = "data"
'''
    
    print("📝 Analyzing code for potential issues...")
    
    try:
        debugger = get_enhanced_debugger()
        insights = debugger.get_predictive_insights(risky_code)
        
        if 'error' in insights:
            print(f"❌ Analysis failed: {insights['error']}")
            return
        
        print("\n🔮 Predictive Analysis Results:")
        print(f"   📊 Health Score: {insights.get('health_score', 0):.1%}")
        print(f"   🎯 Confidence: {insights.get('confidence', 0):.1%}")
        
        # Show risk factors
        risk_factors = insights.get('risk_factors', [])
        if risk_factors:
            print("\n⚠️  Risk Factors Detected:")
            for i, risk in enumerate(risk_factors[:3], 1):
                factor = risk.get('factor', 'unknown')
                description = risk.get('description', 'No description')
                level = risk.get('risk_level', 'unknown')
                print(f"   {i}. {factor.upper()} ({level} risk)")
                print(f"      {description}")
        
        # Show anomalies
        anomalies = insights.get('anomalies', [])
        if anomalies:
            print("\n🚨 Anomalies Found:")
            for i, anomaly in enumerate(anomalies[:3], 1):
                anomaly_type = anomaly.get('type', 'unknown')
                description = anomaly.get('description', 'No description')
                severity = anomaly.get('severity', 'unknown')
                print(f"   {i}. {anomaly_type} ({severity} severity)")
                print(f"      {description}")
        
        # Show predictions
        predictions = insights.get('predictions', {})
        if predictions:
            print("\n🔍 Future Issue Predictions:")
            error_likelihood = predictions.get('error_likelihood', 0)
            security_risk = predictions.get('security_vulnerability', 0)
            maintenance_difficulty = predictions.get('maintenance_difficulty', 0)
            
            print(f"   🐛 Error Likelihood: {error_likelihood:.1%}")
            print(f"   🔒 Security Risk: {security_risk:.1%}")
            print(f"   🔧 Maintenance Difficulty: {maintenance_difficulty:.1%}")
        
        # Show recommendations
        recommendations = insights.get('recommendations', [])
        if recommendations:
            print("\n💡 Proactive Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
    
    except Exception as e:
        print(f"❌ Predictive analysis failed: {e}")
        traceback.print_exc()


def demonstrate_automated_test_generation():
    """Demonstrate automated test generation"""
    print("\n" + "="*80)
    print("🧪 AUTOMATED TEST GENERATION DEMONSTRATION")
    print("="*80)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Test generation not available")
        return
    
    # Sample code for test generation
    test_target_code = '''
def calculate_discount(price, discount_percent, customer_type="regular"):
    """
    Calculate discount amount for a given price
    
    Args:
        price: Original price (must be positive)
        discount_percent: Discount percentage (0-100)
        customer_type: Type of customer ("regular", "premium", "vip")
    
    Returns:
        Discounted price
    
    Raises:
        ValueError: If price is negative or discount is invalid
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount percent must be between 0 and 100")
    
    # Customer type multipliers
    multipliers = {
        "regular": 1.0,
        "premium": 1.2,
        "vip": 1.5
    }
    
    if customer_type not in multipliers:
        raise ValueError(f"Invalid customer type: {customer_type}")
    
    # Calculate discount
    base_discount = price * (discount_percent / 100)
    final_discount = base_discount * multipliers[customer_type]
    
    return price - final_discount

def validate_email(email):
    """Simple email validation"""
    if not email or "@" not in email:
        return False
    
    parts = email.split("@")
    if len(parts) != 2:
        return False
    
    return len(parts[0]) > 0 and len(parts[1]) > 0
'''
    
    print("📝 Generating tests for sample functions...")
    
    try:
        test_results = generate_tests(test_target_code)
        
        if 'error' in test_results:
            print(f"❌ Test generation failed: {test_results['error']}")
            return
        
        test_suite = test_results.get('test_suite', {})
        statistics = test_results.get('statistics', {})
        coverage_report = test_results.get('coverage_report', {})
        
        print("\n🧪 Test Generation Results:")
        print(f"   📊 Total Tests: {statistics.get('total_tests', 0)}")
        print(f"   🎯 Functions Tested: {statistics.get('functions_tested', 0)}")
        print(f"   📈 Average Confidence: {statistics.get('average_confidence', 0):.1%}")
        print(f"   ✅ High Confidence Tests: {statistics.get('high_confidence_tests', 0)}")
        
        # Show test type distribution
        test_distribution = statistics.get('test_type_distribution', {})
        if test_distribution:
            print("\n📋 Test Type Distribution:")
            for test_type, count in test_distribution.items():
                print(f"   • {test_type.replace('_', ' ').title()}: {count}")
        
        # Show coverage report
        print(f"\n📊 Coverage Analysis:")
        print(f"   📈 Coverage: {coverage_report.get('coverage_percentage', 0):.1f}%")
        
        untested = coverage_report.get('untested_functions', [])
        if untested:
            print(f"   ⚠️  Untested Functions: {', '.join(untested)}")
        
        # Show sample generated test
        test_cases = test_suite.get('test_cases', [])
        if test_cases:
            print("\n🔬 Sample Generated Test:")
            sample_test = test_cases[0]
            print(f"   Name: {sample_test.test_name}")
            print(f"   Type: {sample_test.test_type.value}")
            print(f"   Description: {sample_test.description}")
            print(f"   Confidence: {sample_test.confidence:.1%}")
            print("\n   Generated Code:")
            print("   " + "\n   ".join(sample_test.test_code.split("\n")[:10]))
            if len(sample_test.test_code.split("\n")) > 10:
                print("   ... (truncated)")
        
        # Show recommendations
        recommendations = test_results.get('recommendations', [])
        if recommendations:
            print("\n💡 Test Improvement Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
    
    except Exception as e:
        print(f"❌ Test generation failed: {e}")
        traceback.print_exc()


def demonstrate_learning_system():
    """Demonstrate auto-learning capabilities"""
    print("\n" + "="*80)
    print("🎓 AUTO-LEARNING SYSTEM DEMONSTRATION")
    print("="*80)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Learning system not available")
        return
    
    try:
        debugger = get_enhanced_debugger()
        
        # Simulate some debugging sessions for learning
        print("📚 Learning from debugging sessions...")
        
        # Session 1: Attribute Error
        code1 = "result = None\nvalue = result.upper()"
        session1 = debugger.create_debug_session(code1)
        
        try:
            exec(code1)
        except AttributeError as e:
            analysis1 = debugger.analyze_error_with_ml(e, code1)
            # Simulate applying a fix with positive feedback
            fix_suggestions = analysis1.get('fix_suggestions', [])
            if fix_suggestions:
                debugger.apply_fix_suggestion(fix_suggestions[0]['fix_id'], user_feedback=0.9)
        
        # Session 2: Key Error
        code2 = "data = {'a': 1}\nvalue = data['b']"
        session2 = debugger.create_debug_session(code2)
        
        try:
            exec(code2)
        except KeyError as e:
            analysis2 = debugger.analyze_error_with_ml(e, code2)
            fix_suggestions = analysis2.get('fix_suggestions', [])
            if fix_suggestions:
                debugger.apply_fix_suggestion(fix_suggestions[0]['fix_id'], user_feedback=0.8)
        
        # Get learning insights
        learning_insights = debugger.get_learning_insights()
        
        if 'error' not in learning_insights:
            print("\n🧠 Learning System Insights:")
            
            insights = learning_insights.get('insights', {})
            statistics = learning_insights.get('statistics', {})
            
            print(f"   📊 Sessions Analyzed: {statistics.get('total_sessions_learned', 0)}")
            print(f"   🔍 Patterns Discovered: {statistics.get('total_patterns_discovered', 0)}")
            print(f"   🎯 Learning Effectiveness: {statistics.get('learning_effectiveness', 0):.1%}")
            
            # Show top error patterns
            top_patterns = insights.get('top_error_patterns', [])
            if top_patterns:
                print("\n🔍 Top Error Patterns Learned:")
                for i, pattern in enumerate(top_patterns[:3], 1):
                    success_rate = pattern.get('success_rate', 0)
                    occurrences = pattern.get('occurrences', 0)
                    print(f"   {i}. Pattern with {occurrences} occurrences, {success_rate:.1%} success rate")
            
            # Show model update suggestions
            suggestions = learning_insights.get('model_update_suggestions', [])
            if suggestions:
                print("\n💡 Model Improvement Suggestions:")
                for i, suggestion in enumerate(suggestions[:2], 1):
                    suggestion_type = suggestion.get('type', 'unknown')
                    description = suggestion.get('description', 'No description')
                    priority = suggestion.get('priority', 'unknown')
                    print(f"   {i}. {suggestion_type} ({priority} priority)")
                    print(f"      {description}")
        else:
            print(f"❌ Learning insights unavailable: {learning_insights['error']}")
    
    except Exception as e:
        print(f"❌ Learning system demo failed: {e}")
        traceback.print_exc()


def demonstrate_session_management():
    """Demonstrate session management and export"""
    print("\n" + "="*80)
    print("📋 SESSION MANAGEMENT DEMONSTRATION")
    print("="*80)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Session management not available")
        return
    
    try:
        debugger = get_enhanced_debugger()
        
        # Create a comprehensive debug session
        sample_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# This might be slow for large n
result = fibonacci(35)
'''
        
        session_id = debugger.create_debug_session(sample_code, "fibonacci.py")
        print(f"📝 Created debug session: {session_id[:8]}...")
        
        # Perform some analysis
        insights = debugger.get_predictive_insights(sample_code)
        
        # Ask some questions
        debugger.ask_natural_language("How can I optimize this recursive function?")
        debugger.ask_natural_language("What are the performance implications?")
        
        # Generate tests
        debugger.generate_tests_for_session(session_id)
        
        # Get session summary
        summary = debugger.get_session_summary(session_id)
        
        print("\n📊 Session Summary:")
        print(f"   ⏱️  Duration: {summary.get('duration', 0):.1f} seconds")
        print(f"   🔍 Errors Analyzed: {summary.get('error_count', 0)}")
        print(f"   💬 User Interactions: {summary.get('user_interactions', 0)}")
        print(f"   🧠 ML Insights: {'✅' if summary.get('ml_insights_available') else '❌'}")
        print(f"   🧪 Test Results: {'✅' if summary.get('test_results_available') else '❌'}")
        
        health_score = summary.get('code_health_score')
        if health_score is not None:
            print(f"   💚 Code Health: {health_score:.1%}")
        
        # Export session data
        export_result = debugger.export_session_data(session_id)
        
        if 'error' not in export_result:
            print("\n📤 Session Export:")
            print(f"   📊 Format: {export_result.get('format', 'unknown')}")
            print(f"   📏 Size: {export_result.get('size_kb', 0):.1f} KB")
            print("   ✅ Export successful")
        else:
            print(f"   ❌ Export failed: {export_result['error']}")
    
    except Exception as e:
        print(f"❌ Session management demo failed: {e}")
        traceback.print_exc()


def run_comprehensive_demo():
    """Run comprehensive demonstration of all ML debugging features"""
    print("🚀 ABOV3 Genesis - Enhanced ML Debug Engine")
    print("Claude-level Intelligent Debugging System Demonstration")
    print("=" * 80)
    
    if not ENHANCED_AVAILABLE:
        print("⚠️  Enhanced ML debugging features are not fully available.")
        print("This may be due to missing dependencies or import issues.")
        print("The demonstration will show what's possible when fully configured.\n")
    
    # Run all demonstrations
    demonstrations = [
        ("ML Error Analysis", demonstrate_ml_error_analysis),
        ("Natural Language Interface", demonstrate_natural_language_interface),
        ("Predictive Debugging", demonstrate_predictive_debugging),
        ("Automated Test Generation", demonstrate_automated_test_generation),
        ("Auto-Learning System", demonstrate_learning_system),
        ("Session Management", demonstrate_session_management),
    ]
    
    for name, demo_func in demonstrations:
        try:
            demo_func()
        except KeyboardInterrupt:
            print(f"\n⚠️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ {name} demonstration failed: {e}")
    
    print("\n" + "="*80)
    print("🎉 DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThe ABOV3 Genesis Enhanced ML Debug Engine provides:")
    print("• 🧠 Transformer-based error pattern recognition")
    print("• 🔮 Predictive debugging to catch issues before they occur")
    print("• 💬 Natural language debugging interface")
    print("• 🧪 Automated comprehensive test generation")
    print("• 🎓 Auto-learning system that improves over time")
    print("• 📊 Detailed session management and analytics")
    print("\nThis represents Claude-level intelligence for software debugging!")


if __name__ == "__main__":
    # Check if we should run a specific demo
    if len(sys.argv) > 1:
        demo_name = sys.argv[1].lower()
        demo_functions = {
            'error': demonstrate_ml_error_analysis,
            'nl': demonstrate_natural_language_interface,
            'predictive': demonstrate_predictive_debugging,
            'tests': demonstrate_automated_test_generation,
            'learning': demonstrate_learning_system,
            'session': demonstrate_session_management,
        }
        
        if demo_name in demo_functions:
            print(f"Running {demo_name} demonstration...")
            demo_functions[demo_name]()
        else:
            print(f"Unknown demo: {demo_name}")
            print(f"Available demos: {', '.join(demo_functions.keys())}")
    else:
        # Run comprehensive demo
        run_comprehensive_demo()