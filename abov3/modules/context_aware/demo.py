"""
ABOV3 Genesis - Context-Aware Comprehension Demo
Demonstration of the context-aware comprehension capabilities
"""

import asyncio
import logging
import time
from pathlib import Path
import json
from typing import Dict, List, Any

from .core.comprehension_engine import ComprehensionEngine, ComprehensionRequest, ComprehensionMode
from ..nl2code.utils.ollama_integration import OllamaIntegration
from ...core.ollama_client import OllamaClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextAwareDemo:
    """
    Demo class for context-aware comprehension features
    """
    
    def __init__(self, workspace_path: str, model_name: str = "deepseek-coder"):
        self.workspace_path = Path(workspace_path)
        self.model_name = model_name
        self.ollama_client = None
        self.comprehension_engine = None
        
        logger.info(f"Initialized ContextAwareDemo for workspace: {workspace_path}")
    
    async def initialize(self):
        """Initialize the demo environment"""
        logger.info("Initializing Context-Aware Comprehension Demo...")
        
        try:
            # Initialize Ollama client
            self.ollama_client = OllamaClient()
            await self.ollama_client.connect()
            
            # Check if model is available
            models = await self.ollama_client.list_models()
            available_models = [m['name'] for m in models]
            
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                if available_models:
                    self.model_name = available_models[0]
                    logger.info(f"Using available model: {self.model_name}")
                else:
                    raise Exception("No Ollama models available")
            
            # Initialize comprehension engine
            self.comprehension_engine = ComprehensionEngine(
                workspace_path=str(self.workspace_path),
                ollama_client=self.ollama_client,
                model_name=self.model_name
            )
            
            await self.comprehension_engine.initialize()
            
            logger.info("Context-Aware Comprehension Demo initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize demo: {e}")
            raise
    
    async def run_comprehensive_demo(self):
        """Run a comprehensive demo of all features"""
        if not self.comprehension_engine:
            await self.initialize()
        
        print("\\n" + "="*80)
        print("🧠 ABOV3 CONTEXT-AWARE COMPREHENSION DEMO")
        print("="*80)
        
        # Demo scenarios
        await self._demo_quick_scan()
        await self._demo_deep_analysis()
        await self._demo_semantic_search()
        await self._demo_qa_mode()
        await self._demo_refactoring_suggestions()
        await self._demo_monorepo_analysis()
        await self._demo_performance_stats()
    
    async def _demo_quick_scan(self):
        """Demo quick scan functionality"""
        print("\\n🔍 QUICK SCAN DEMO")
        print("-" * 40)
        
        request = ComprehensionRequest(
            query="What are the main components of this codebase?",
            mode=ComprehensionMode.QUICK_SCAN,
            max_files=20
        )
        
        print(f"Query: {request.query}")
        print("Mode: Quick Scan")
        
        start_time = time.time()
        result = await self.comprehension_engine.comprehend(request)
        duration = time.time() - start_time
        
        print(f"\\nResult (completed in {duration:.2f}s):")
        print(f"Answer: {result.answer[:500]}..." if len(result.answer) > 500 else result.answer)
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Files analyzed: {len(result.source_files)}")
        print(f"Cache hit: {result.cache_hit}")
    
    async def _demo_deep_analysis(self):
        """Demo deep analysis functionality"""
        print("\\n🔬 DEEP ANALYSIS DEMO")
        print("-" * 40)
        
        request = ComprehensionRequest(
            query="Analyze the architecture and design patterns in this codebase",
            mode=ComprehensionMode.DEEP_ANALYSIS,
            max_files=50,
            context_depth=3
        )
        
        print(f"Query: {request.query}")
        print("Mode: Deep Analysis")
        
        start_time = time.time()
        result = await self.comprehension_engine.comprehend(request)
        duration = time.time() - start_time
        
        print(f"\\nResult (completed in {duration:.2f}s):")
        print(f"Answer: {result.answer[:500]}..." if len(result.answer) > 500 else result.answer)
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Files analyzed: {len(result.source_files)}")
        print(f"Related concepts: {', '.join(result.related_concepts[:5])}")
        
        if result.metadata:
            print(f"Additional metadata: {result.metadata}")
    
    async def _demo_semantic_search(self):
        """Demo semantic search functionality"""
        print("\\n🔎 SEMANTIC SEARCH DEMO")
        print("-" * 40)
        
        request = ComprehensionRequest(
            query="Find code patterns for error handling and exception management",
            mode=ComprehensionMode.SEMANTIC_SEARCH,
            max_files=30
        )
        
        print(f"Query: {request.query}")
        print("Mode: Semantic Search")
        
        start_time = time.time()
        result = await self.comprehension_engine.comprehend(request)
        duration = time.time() - start_time
        
        print(f"\\nResult (completed in {duration:.2f}s):")
        print(f"Answer: {result.answer[:500]}..." if len(result.answer) > 500 else result.answer)
        print(f"Files with matches: {len(result.source_files)}")
        
        if result.metadata and 'search_results' in result.metadata:
            print(f"Search results found: {result.metadata['search_results']}")
    
    async def _demo_qa_mode(self):
        """Demo Q&A mode functionality"""
        print("\\n❓ Q&A MODE DEMO")
        print("-" * 40)
        
        questions = [
            "How does the Ollama client handle connections?",
            "What design patterns are used in the assistant module?",
            "Where are the main configuration settings defined?"
        ]
        
        for question in questions:
            print(f"\\nQuestion: {question}")
            
            request = ComprehensionRequest(
                query=question,
                mode=ComprehensionMode.QA_MODE,
                max_files=15
            )
            
            start_time = time.time()
            result = await self.comprehension_engine.comprehend(request)
            duration = time.time() - start_time
            
            print(f"Answer ({duration:.2f}s): {result.answer[:300]}...")
            
            if result.metadata and 'intent' in result.metadata:
                print(f"Detected intent: {result.metadata['intent']}")
    
    async def _demo_refactoring_suggestions(self):
        """Demo refactoring suggestions"""
        print("\\n🔧 REFACTORING SUGGESTIONS DEMO")
        print("-" * 40)
        
        request = ComprehensionRequest(
            query="What refactoring improvements can be made to this codebase?",
            mode=ComprehensionMode.REFACTOR_MODE,
            max_files=20
        )
        
        print(f"Query: {request.query}")
        print("Mode: Refactoring Analysis")
        
        start_time = time.time()
        result = await self.comprehension_engine.comprehend(request)
        duration = time.time() - start_time
        
        print(f"\\nResult (completed in {duration:.2f}s):")
        print(f"Analysis: {result.answer[:400]}...")
        print(f"\\nSuggestions:")
        for i, suggestion in enumerate(result.suggestions[:5], 1):
            print(f"  {i}. {suggestion}")
        
        if result.metadata:
            print(f"\\nRefactoring metadata: {result.metadata}")
    
    async def _demo_monorepo_analysis(self):
        """Demo monorepo analysis"""
        print("\\n🏗️  MONOREPO ANALYSIS DEMO")
        print("-" * 40)
        
        request = ComprehensionRequest(
            query="Provide an overview of this monorepo structure and key modules",
            mode=ComprehensionMode.MONOREPO_MODE,
            max_files=100,
            context_depth=2
        )
        
        print(f"Query: {request.query}")
        print("Mode: Monorepo Analysis")
        
        start_time = time.time()
        result = await self.comprehension_engine.comprehend(request)
        duration = time.time() - start_time
        
        print(f"\\nResult (completed in {duration:.2f}s):")
        print(f"Overview: {result.answer[:400]}...")
        print(f"Key modules analyzed: {len(result.related_concepts)}")
        
        if result.metadata:
            print(f"Modules in scope: {result.metadata.get('modules_analyzed', 'N/A')}")
            print(f"Files in scope: {result.metadata.get('files_in_scope', 'N/A')}")
    
    async def _demo_performance_stats(self):
        """Demo performance statistics"""
        print("\\n📊 PERFORMANCE STATISTICS")
        print("-" * 40)
        
        stats = self.comprehension_engine.get_performance_stats()
        
        print(f"Total queries processed: {stats['total_queries']}")
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"Average response time: {stats['avg_response_time']:.3f}s")
        print(f"Files indexed: {stats['total_files_indexed']}")
        print(f"Cache size: {stats['cache_size']} items")
        
        if 'last_index_update' in stats and stats['last_index_update']:
            print(f"Last index update: {time.ctime(stats['last_index_update'])}")
    
    async def run_interactive_demo(self):
        """Run interactive demo where user can ask questions"""
        if not self.comprehension_engine:
            await self.initialize()
        
        print("\\n" + "="*60)
        print("🤖 INTERACTIVE CONTEXT-AWARE Q&A")
        print("="*60)
        print("Ask questions about your codebase!")
        print("Type 'quit', 'exit', or 'q' to stop")
        print("Type 'help' for available modes")
        print("-"*60)
        
        while True:
            try:
                # Get user input
                print("\\n💬 Your question:")
                user_query = input("> ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("\\n👋 Thanks for trying ABOV3 Context-Aware Comprehension!")
                    break
                
                if user_query.lower() == 'help':
                    self._print_help()
                    continue
                
                if not user_query:
                    continue
                
                # Detect mode from query
                mode = self._detect_query_mode(user_query)
                
                # Create request
                request = ComprehensionRequest(
                    query=user_query,
                    mode=mode,
                    max_files=30
                )
                
                print(f"\\n🔍 Processing ({mode.value})...")
                start_time = time.time()
                
                # Get response
                result = await self.comprehension_engine.comprehend(request)
                duration = time.time() - start_time
                
                # Display result
                print(f"\\n🧠 Response ({duration:.2f}s):")
                print("-" * 40)
                print(result.answer)
                
                if result.suggestions:
                    print("\\n💡 Suggestions:")
                    for i, suggestion in enumerate(result.suggestions[:3], 1):
                        print(f"  {i}. {suggestion}")
                
                print(f"\\n📈 Stats: {result.confidence_score:.1%} confidence | "
                      f"{len(result.source_files)} files | "
                      f"{'Cache hit' if result.cache_hit else 'Fresh analysis'}")
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\\n❌ Error: {e}")
                logger.error(f"Interactive demo error: {e}")
    
    def _detect_query_mode(self, query: str) -> ComprehensionMode:
        """Detect appropriate comprehension mode from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['find', 'search', 'pattern', 'similar']):
            return ComprehensionMode.SEMANTIC_SEARCH
        elif any(word in query_lower for word in ['refactor', 'improve', 'optimize', 'clean']):
            return ComprehensionMode.REFACTOR_MODE
        elif any(word in query_lower for word in ['overview', 'summary', 'quick']):
            return ComprehensionMode.QUICK_SCAN
        elif any(word in query_lower for word in ['architecture', 'analyze', 'deep', 'detailed']):
            return ComprehensionMode.DEEP_ANALYSIS
        elif any(word in query_lower for word in ['monorepo', 'structure', 'modules']):
            return ComprehensionMode.MONOREPO_MODE
        else:
            return ComprehensionMode.QA_MODE
    
    def _print_help(self):
        """Print help information"""
        print("\\n📖 HELP - Available Query Types:")
        print("-" * 40)
        print("🔍 Quick questions: 'What does this code do?'")
        print("🔬 Deep analysis: 'Analyze the architecture of this project'")
        print("🔎 Semantic search: 'Find error handling patterns'")
        print("❓ Q&A: 'How does the authentication work?'")
        print("🔧 Refactoring: 'What can be improved in this code?'")
        print("🏗️  Monorepo: 'Give me an overview of this repository structure'")
        print("-" * 40)
        print("💡 Tips:")
        print("  - Be specific about what you want to know")
        print("  - Ask about patterns, architecture, or specific functionality")
        print("  - Request suggestions for improvements or optimizations")
    
    async def save_demo_results(self, output_file: str = "context_aware_demo_results.json"):
        """Save demo results for analysis"""
        if not self.comprehension_engine:
            return
        
        demo_data = {
            'timestamp': time.time(),
            'workspace_path': str(self.workspace_path),
            'model_name': self.model_name,
            'performance_stats': self.comprehension_engine.get_performance_stats(),
            'demo_completed': True
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(demo_data, f, indent=2)
        
        print(f"\\n💾 Demo results saved to: {output_path.absolute()}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.comprehension_engine:
                await self.comprehension_engine.clear_cache()
            
            if self.ollama_client:
                await self.ollama_client.close()
            
            logger.info("Demo cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Main demo functions
async def run_full_demo(workspace_path: str, model_name: str = "deepseek-coder"):
    """Run the full demonstration"""
    demo = ContextAwareDemo(workspace_path, model_name)
    
    try:
        await demo.initialize()
        await demo.run_comprehensive_demo()
        await demo.save_demo_results()
    finally:
        await demo.cleanup()

async def run_interactive_session(workspace_path: str, model_name: str = "deepseek-coder"):
    """Run interactive Q&A session"""
    demo = ContextAwareDemo(workspace_path, model_name)
    
    try:
        await demo.initialize()
        await demo.run_interactive_demo()
        await demo.save_demo_results()
    finally:
        await demo.cleanup()

if __name__ == "__main__":
    import sys
    
    # Get workspace path from command line or use current directory
    workspace = sys.argv[1] if len(sys.argv) > 1 else "."
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-coder"
    
    print("🚀 Starting ABOV3 Context-Aware Comprehension Demo")
    print(f"📁 Workspace: {Path(workspace).absolute()}")
    print(f"🤖 Model: {model}")
    
    # Ask user for demo type
    print("\\nSelect demo type:")
    print("1. Full comprehensive demo")
    print("2. Interactive Q&A session")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_full_demo(workspace, model))
    else:
        asyncio.run(run_interactive_session(workspace, model))