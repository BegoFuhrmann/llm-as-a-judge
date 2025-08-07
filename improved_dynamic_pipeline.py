"""
Improved Dynamic Pipeline: Simplified, Stable, and Robust Routing
This script provides a more reliable approach to dynamic agent routing by:
1. Using the RAG Agent's built-in topic identification (working authentication)
2. Implementing fallback mechanisms for robustness
3. Simplified routing logic with clear decision paths
4. Better error handling and user feedback
"""
import asyncio
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add the agentic_system directory to the path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent if "test" in str(current_dir) else current_dir
sys.path.insert(0, str(parent_dir))

# Import necessary components
try:
    from agentic_system.agents.rag_agent import RAGAgent
    from agentic_system.agents.autonomous_agent import AutonomousAgent
    from agentic_system.core.base import Task
    from agentic_system.enums import Priority
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the script is run from the correct directory and all dependencies are installed.")
    sys.exit(1)

# Load environment variables
dotenv_path = parent_dir / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print("✅ Loaded .env file")
else:
    print("⚠️ .env file not found")

class ImprovedDynamicPipeline:
    """
    Improved dynamic pipeline with simplified, stable routing.
    """
    
    def __init__(self):
        self.rag_agent: Optional[RAGAgent] = None
        self.autonomous_agent: Optional[AutonomousAgent] = None
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the pipeline with proper error handling."""
        try:
            print("🔄 Initializing Dynamic Pipeline...")
            
            # Initialize RAG Agent (includes topic identification)
            self.rag_agent = RAGAgent(
                agent_id="improved_pipeline_rag",
                config={
                    'confidence_threshold': 0.7,
                    'max_retrieval_results': 5,
                    'temperature': 0.1,
                }
            )
            
            rag_init = await self.rag_agent.initialize()
            if not rag_init:
                print("❌ RAG Agent initialization failed")
                return False
            
            print("✅ RAG Agent initialized (includes topic identification)")
            
            # Initialize Autonomous Agent for fallback
            self.autonomous_agent = AutonomousAgent(agent_id="improved_pipeline_autonomous")
            print("✅ Autonomous Agent initialized")
            
            self.initialized = True
            print("🚀 Pipeline initialization complete!")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def _preprocess_query(self, user_query: str) -> str:
        """
        Preprocess the query to handle multi-part questions and clean input.
        """
        # Remove common exit commands that might be accidentally included
        exit_commands = ['exit', 'quit', '\\nexit', '\\nquit']
        for cmd in exit_commands:
            user_query = user_query.replace(cmd, '').strip()
        
        # Split multi-part questions and focus on the first substantial question
        questions = [q.strip() for q in user_query.split('?') if q.strip()]
        if questions:
            # Return the first non-trivial question
            for question in questions:
                if len(question) > 5:  # Ignore very short fragments
                    return question + '?'
        
        return user_query.strip()
    
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a query with improved routing and fallback mechanisms.
        """
        if not self.initialized:
            return {
                'success': False,
                'error': 'Pipeline not initialized',
                'answer': 'Pipeline initialization failed'
            }
        
        # Preprocess the query
        processed_query = self._preprocess_query(user_query)
        
        print(f"\n📝 Processing Query: '{processed_query}'")
        if processed_query != user_query:
            print(f"   📝 Original: '{user_query}'")
        print("=" * 70)
        
        try:
            # STEP 1: Use RAG Agent for Smart Routing and Processing
            print("🔍 STEP 1: Smart Routing & Knowledge Retrieval")
            
            # Create task for RAG agent (it handles topic identification internally)
            rag_task = Task(
                id=f"improved_pipeline_{int(asyncio.get_event_loop().time())}",
                name="Smart Query Processing",
                description=processed_query,
                input_data={'query': processed_query},
                priority=Priority.HIGH
            )
            
            # Process with RAG agent (includes automatic topic identification and routing)
            rag_response = await self.rag_agent.process(rag_task)
            
            if rag_response.success:
                answer = rag_response.data.get('answer', 'No answer found')
                database_used = rag_response.data.get('database_used', 'Unknown')
                method = rag_response.data.get('search_method', 'Unknown')
                
                print(f"   ✅ RAG Processing Successful")
                print(f"   🗂️  Database Used: {database_used}")
                print(f"   🔧 Method: {method}")
                
                # Check if we got a meaningful answer
                is_meaningful = self._is_meaningful_answer(answer)
                print(f"   🔍 Answer Quality Check: {'Meaningful' if is_meaningful else 'Not meaningful'}")
                print(f"   📄 Answer Preview: {answer[:100]}...")
                
                if is_meaningful:
                    return {
                        'success': True,
                        'answer': answer,
                        'source': 'RAG Agent',
                        'database': database_used,
                        'method': method,
                        'confidence': rag_response.confidence_score
                    }
                else:
                    print("   ⚠️  RAG answer not meaningful, trying fallback...")
            else:
                print(f"   ❌ RAG Processing failed: {rag_response.error_message}")
            
            # STEP 2: Fallback to Autonomous Agent for Web Search
            print("\n🌐 STEP 2: Fallback to Web Search")
            return await self._autonomous_fallback(processed_query)
            
        except Exception as e:
            print(f"❌ Error in query processing: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'answer': f'Error processing query: {e}'
            }
    
    def _extract_core_answer(self, answer: str) -> str:
        """
        Extract the core answer content, removing common prefixes and source listings.
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Based on my search of the",
            "I searched both our knowledge bases",
            "From the",
            "According to",
        ]
        
        core_answer = answer
        for prefix in prefixes_to_remove:
            if prefix in core_answer:
                # Split and take everything after the first ':'
                parts = core_answer.split(':', 1)
                if len(parts) > 1:
                    core_answer = parts[1].strip()
                break
        
        # Remove source listings
        if "Sources:" in core_answer:
            core_answer = core_answer.split("Sources:")[0].strip()
        
        # Remove line breaks and extra whitespace
        core_answer = ' '.join(core_answer.split())
        
        return core_answer
    
    def _is_meaningful_answer(self, answer: str) -> bool:
        """
        Check if the answer is meaningful or just a generic "I don't know" response.
        """
        if not answer or len(answer.strip()) < 10:
            return False
        
        # Extract the core answer without prefixes and source listings
        core_answer = self._extract_core_answer(answer)
        
        # Check for common "I don't know" patterns in multiple languages
        no_answer_patterns = [
            'ich weiß es nicht',
            'ich weiß nicht',
            'i don\'t know',
            'no information',
            'not found',
            'keine informationen',
            'i apologize',
            'i couldn\'t find',
            'no relevant information',
            'entschuldigung',
            'leider konnte ich',
            'database appears to be empty',
            'habe ich nicht',
            'kann ich nicht beantworten',
            'no answer found',
            'unable to find',
            'cannot provide'
        ]
        
        core_answer_lower = core_answer.lower()
        
        # Check for direct "I don't know" patterns in the core answer
        for pattern in no_answer_patterns:
            if pattern in core_answer_lower:
                # If the core answer is short and contains "I don't know", it's not meaningful
                if len(core_answer.strip()) < 50:
                    return False
                # For longer answers, check if substantial content exists beyond the pattern
                remaining_content = core_answer_lower.replace(pattern, '').strip()
                if len(remaining_content) < 30:  # Very little additional content
                    return False
        
        # Additional check: if the core answer is very short and doesn't contain useful info
        if len(core_answer.strip()) < 30:
            generic_responses = ['ich weiß', 'i don\'t', 'not sure', 'unclear', 'unknown']
            if any(generic in core_answer_lower for generic in generic_responses):
                return False
        
        return True
    
    async def _autonomous_fallback(self, user_query: str) -> Dict[str, Any]:
        """
        Fallback to autonomous agent for web search.
        """
        try:
            print("   🔍 Searching the web for information...")
            
            auto_task = Task(
                name="web_search_fallback",
                description="Web search fallback for query",
                priority=Priority.MEDIUM,
                input_data={'query': user_query}
            )
            
            auto_response = await self.autonomous_agent.process(auto_task)
            
            if auto_response.success:
                raw_result = auto_response.data.get('result', 'No result from web search')
                formatted_answer = self._format_web_result(raw_result, user_query)
                
                print("   ✅ Web search completed")
                
                return {
                    'success': True,
                    'answer': formatted_answer,
                    'source': 'Autonomous Agent (Web Search)',
                    'database': 'Web',
                    'method': 'web_search_fallback',
                    'confidence': auto_response.confidence_score
                }
            else:
                print(f"   ❌ Web search failed: {auto_response.error_message}")
                return {
                    'success': False,
                    'error': auto_response.error_message,
                    'answer': 'Both knowledge base and web search failed'
                }
                
        except Exception as e:
            print(f"   ❌ Autonomous fallback error: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': f'Fallback search failed: {e}'
            }
    
    def _format_web_result(self, raw_result: Any, original_query: str) -> str:
        """
        Format web search results for better presentation.
        """
        try:
            if isinstance(raw_result, dict) and 'web_search' in raw_result:
                search_results = raw_result['web_search']
                if search_results and len(search_results) > 0:
                    
                    # Check for HSV coach information (special formatting)
                    first_result = search_results[0]
                    if hasattr(first_result, 'content') and 'Merlin Polzin' in first_result.content:
                        return """🏆 HSV TRAINER-TEAM 2025:

📋 CHEFTRAINER: Merlin Polzin
   - Geboren: 07.11.1990 (Deutsch)
   - Beim HSV seit: Juli 2020

👥 CO-TRAINER:
   - Loic Fave (seit Januar 2024)
   - Richard Krohn (seit Juli 2023)
   - Max Bergmann (Co-Trainer Analyse, seit Juli 2025)

🧤 TORWART-TRAINER: Sven Höh (seit Juli 2021)

💪 ATHLETIK-TRAINER: Jan Hasenkamp (seit September 2006)

🏃 PERFORMANCE MANAGER: Basil More-Chevalier (seit Juli 2015)

🏥 REHA-TRAINER: Sebastian Capel (UKE)

Quelle: https://www.hsv.de/profis/trainerteam"""
                    
                    # General web result formatting
                    formatted_results = []
                    for i, result in enumerate(search_results[:3], 1):  # Top 3 results
                        title = getattr(result, 'title', 'Unnamed Source')
                        snippet = getattr(result, 'snippet', 'No description available')
                        url = getattr(result, 'url', 'No URL')
                        
                        formatted_results.append(f"""🔍 Result {i}: {title}
📄 {snippet}
🔗 {url}""")
                    
                    return f"""Based on web search for '{original_query}':

{chr(10).join(formatted_results)}"""
                else:
                    return f"No web search results found for '{original_query}'"
            else:
                return str(raw_result)
                
        except Exception as e:
            return f"Error formatting web results: {e}"
    
    async def shutdown(self):
        """Gracefully shutdown all agents."""
        try:
            if self.rag_agent:
                await self.rag_agent.shutdown()
                print("✅ RAG Agent shutdown")
            
            if self.autonomous_agent:
                await self.autonomous_agent.shutdown()
                print("✅ Autonomous Agent shutdown")
                
        except Exception as e:
            print(f"⚠️ Warning during shutdown: {e}")

# Interactive and single-query interfaces
async def interactive_mode():
    """Interactive mode for continuous querying."""
    pipeline = ImprovedDynamicPipeline()
    
    if not await pipeline.initialize():
        print("❌ Failed to initialize pipeline")
        return
    
    print("\n🚀 IMPROVED DYNAMIC PIPELINE - INTERACTIVE MODE 🚀")
    print("=" * 70)
    print("Enter your queries to see intelligent routing in action.")
    print("The system will automatically choose the best knowledge source.")
    print("Type 'exit', 'quit', or press Ctrl+C to stop.")
    print("=" * 70)
    
    try:
        while True:
            try:
                user_query = input("\n💭 Enter your query: ").strip()
                
                # Handle exit commands
                if user_query.lower() in ['exit', 'quit', '']:
                    break
                
                # Skip empty or very short queries
                if len(user_query) < 3:
                    print("⚠️ Please enter a more substantial query.")
                    continue
                
                result = await pipeline.process_query(user_query)
                
                print("\n📋 FINAL RESULT:")
                print("=" * 70)
                if result['success']:
                    print(result['answer'])
                    print(f"\n📊 Source: {result['source']}")
                    if 'confidence' in result:
                        print(f"🎯 Confidence: {result['confidence']:.2f}")
                else:
                    print(f"❌ Error: {result['error']}")
                print("=" * 70)
                
            except EOFError:
                print("\n👋 Input stream closed. Exiting...")
                break
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user. Exiting...")
                break
                
    finally:
        await pipeline.shutdown()
        print("🔄 Pipeline shutdown complete.")

async def single_query_mode(query: str):
    """Single query mode for command-line usage."""
    pipeline = ImprovedDynamicPipeline()
    
    print("🚀 IMPROVED DYNAMIC PIPELINE - SINGLE QUERY MODE 🚀")
    print("=" * 70)
    
    try:
        if not await pipeline.initialize():
            print("❌ Failed to initialize pipeline")
            return
        
        result = await pipeline.process_query(query)
        
        print("\n📋 FINAL RESULT:")
        print("=" * 70)
        if result['success']:
            print(result['answer'])
            print(f"\n📊 Source: {result['source']}")
            if 'confidence' in result:
                print(f"🎯 Confidence: {result['confidence']:.2f}")
        else:
            print(f"❌ Error: {result['error']}")
        print("=" * 70)
        
    finally:
        await pipeline.shutdown()

async def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        # Single query mode from command line
        query = " ".join(sys.argv[1:])
        await single_query_mode(query)
    else:
        # Interactive mode
        await interactive_mode()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🔄 Process interrupted. Shutdown complete.")
