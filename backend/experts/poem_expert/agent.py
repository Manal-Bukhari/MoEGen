"""
Poem Expert LangGraph Agent
Placeholder for future LangGraph agent implementation.
"""
# TODO: Implement LangGraph agent for poetry generation
# This will use Gemini via LangChain to generate creative poetry

"""
Poem Expert LangGraph Agent

This is the MAIN FILE that controls poem generation.

HOW IT WORKS:
1. User request comes in → agent.py receives it
2. Extract Context: Understand what poem to write (type, tone, theme)
3. Generate Poem: Use Gemini AI to create the poem
4. Evaluate (optional): Check if poem is good quality
5. Regenerate (optional): If score is low, try again with feedback

LANGGRAPH EXPLANATION:
- StateGraph = A workflow with steps (nodes) connected by arrows (edges)
- State = Shared memory (dictionary) that all steps can read/write
- Nodes = Individual steps (functions)
- Edges = Connections between steps
"""

import logging
from typing import Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import our configuration
from .config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    USE_EVALUATOR,
    EVALUATOR_THRESHOLD,
    EVALUATOR_MAX_RETRIES,
    DEFAULT_POEM_TYPE,
    DEFAULT_TONE
)

# Import our prompt templates
from .prompts import POEM_SYSTEM_PROMPT, POEM_USER_PROMPT_TEMPLATE

# Import our tools
from .tools import ContextExtractor, PoemEvaluator

logger = logging.getLogger(__name__)


# ============================================
# STEP 1: DEFINE STATE (Shared Memory)
# ============================================
class PoemAgentState(TypedDict):
    """
    This is the SHARED MEMORY that flows through all steps.
    
    Think of it as a notebook that each step can read and write to.
    
    Fields:
    - prompt: What the user asked for
    - enhanced_query: Extra info from the router
    - extracted_context: What kind of poem to write
    - generated_poem: The poem we created
    - evaluation: Quality score (if evaluator is enabled)
    - attempt: How many times we've tried to regenerate
    - final_poem: The final output
    - max_length: Token limit
    - temperature: How creative the AI should be
    - max_tokens_retry_count: Counter for handling truncation
    """
    prompt: str
    enhanced_query: Optional[Dict[str, Any]]
    extracted_context: Optional[Dict[str, Any]]
    generated_poem: Optional[str]
    evaluation: Optional[Dict[str, Any]]
    attempt: int
    final_poem: str
    max_length: Optional[int]
    temperature: Optional[float]
    max_tokens_retry_count: int


# ============================================
# STEP 2: CREATE THE AGENT CLASS
# ============================================
class PoemExpertAgent:
    """
    This is the MAIN AGENT CLASS.
    
    It's like a manager that:
    1. Initializes all the tools (context extractor, evaluator)
    2. Sets up the AI model (Gemini)
    3. Builds the workflow (LangGraph)
    4. Runs the workflow when asked
    """
    
    def __init__(self):
        """
        Initialize the agent.
        
        This runs ONCE when the server starts.
        It sets up everything we need.
        """
        logger.info("🚀 Initializing Poem Expert Agent...")
        
        # Initialize our tools
        self.context_extractor = ContextExtractor(api_key=GEMINI_API_KEY)
        self.poem_evaluator = PoemEvaluator(
            api_key=GEMINI_API_KEY,
            threshold=EVALUATOR_THRESHOLD,
            max_retries=EVALUATOR_MAX_RETRIES
        )
        
        # Initialize Gemini (the AI that generates poems)
        if GEMINI_API_KEY:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GEMINI_API_KEY,
                    temperature=TEMPERATURE,  # Higher = more creative
                    max_output_tokens=MAX_TOKENS
                )
                logger.info(f"✅ Gemini LLM initialized: {GEMINI_MODEL}")
            except Exception as e:
                logger.error(f"❌ LLM initialization failed: {e}")
                self.llm = None
        else:
            logger.warning("⚠️ No GEMINI_API_KEY found in .env")
            self.llm = None
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
        logger.info("✅ Poem Expert Agent ready!")
    
    # ============================================
    # STEP 3: BUILD THE WORKFLOW (LangGraph)
    # ============================================
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        This defines the FLOW of our agent:
        
        START
          ↓
        Extract Context (understand requirements)
          ↓
        Generate Poem (create using AI)
          ↓
        Evaluate Poem (check quality - optional)
          ↓
        Is it good?
          ├─ YES → END
          └─ NO → Regenerate (with feedback) → Generate Poem
        
        LANGGRAPH CONCEPTS:
        - Node = A step (function) in the workflow
        - Edge = Connection between nodes
        - Conditional Edge = A decision point (if/else)
        """
        workflow = StateGraph(PoemAgentState)
        
        # Add nodes (steps)
        workflow.add_node("extract_context", self._extract_context_node)
        workflow.add_node("generate_poem", self._generate_poem_node)
        workflow.add_node("evaluate_poem", self._evaluate_poem_node)
        workflow.add_node("regenerate_if_needed", self._regenerate_if_needed_node)
        
        # Set entry point (where to start)
        workflow.set_entry_point("extract_context")
        
        # Add edges (connections)
        workflow.add_edge("extract_context", "generate_poem")
        workflow.add_edge("generate_poem", "evaluate_poem")
        
        # Add conditional edge (decision point)
        workflow.add_conditional_edges(
            "evaluate_poem",  # After evaluating...
            self._should_regenerate,  # ...call this function to decide...
            {
                "regenerate": "regenerate_if_needed",  # ...go here if bad
                "end": END  # ...or end if good
            }
        )
        
        # Regenerate loops back to generate
        workflow.add_edge("regenerate_if_needed", "generate_poem")
        
        # Compile and return
        return workflow.compile()
    
    # ============================================
    # STEP 4: DEFINE NODES (Workflow Steps)
    # ============================================
    
    def _extract_context_node(self, state: PoemAgentState) -> PoemAgentState:
        """
        NODE 1: Extract Context
        
        PURPOSE: Understand what kind of poem to write
        
        WHAT IT DOES:
        1. Takes user's prompt
        2. Calls ContextExtractor tool
        3. Extracts: poem type, tone, theme, rhyme scheme
        4. Stores in state["extracted_context"]
        
        EXAMPLE:
        Input: "Write a sad poem about rain"
        Output: {
            "poem_type": "free_verse",
            "tone": "melancholic",
            "theme": "rain",
            "rhyme_scheme": "free_verse"
        }
        """
        logger.info("🔍 NODE 1: Extracting context...")
        
        try:
            # Call the context extractor tool
            extracted_context = self.context_extractor.extract(
                state["prompt"],
                state.get("enhanced_query")
            )
            
            # Store in state
            state["extracted_context"] = extracted_context
            
            logger.info(f"✅ Context extracted:")
            logger.info(f"   Poem Type: {extracted_context.get('poem_type')}")
            logger.info(f"   Tone: {extracted_context.get('tone')}")
            logger.info(f"   Theme: {extracted_context.get('theme')}")
            
        except Exception as e:
            logger.error(f"❌ Context extraction failed: {e}")
            
            # Fallback: Use defaults
            state["extracted_context"] = {
                "poem_type": DEFAULT_POEM_TYPE,
                "tone": DEFAULT_TONE,
                "theme": "general",
                "rhyme_scheme": "free_verse",
                "special_requirements": []
            }
            logger.warning("⚠️ Using default context")
        
        return state
    
    def _generate_poem_node(self, state: PoemAgentState) -> PoemAgentState:
        """
        NODE 2: Generate Poem
        
        PURPOSE: Create the actual poem using Gemini AI
        
        WHAT IT DOES:
        1. Takes extracted context from state
        2. Builds a detailed prompt for Gemini
        3. Calls Gemini API to generate poem
        4. Handles truncation (if poem is cut off)
        5. Stores poem in state["generated_poem"]
        """
        logger.info("✍️ NODE 2: Generating poem...")
        
        # Check if LLM is available
        if not self.llm:
            logger.error("❌ LLM not initialized")
            state["generated_poem"] = "Error: Gemini API not available"
            return state
        
        try:
            # Get data from state
            extracted_context = state.get("extracted_context", {})
            attempt = state.get("attempt", 0)
            
            # Get original query (remove feedback if regenerating)
            current_prompt = state.get("prompt", "")
            original_query = current_prompt
            feedback_section = ""
            
            if "=== FEEDBACK" in current_prompt:
                parts = current_prompt.split("=== FEEDBACK", 1)
                original_query = parts[0].strip()
                feedback_section = "\n\n=== FEEDBACK" + parts[1] if len(parts) > 1 else ""
            
            # Build the prompt for Gemini
            user_prompt = POEM_USER_PROMPT_TEMPLATE.format(
                enhanced_instruction=state.get("enhanced_query", {}).get("enhanced_instruction", original_query),
                poem_type=extracted_context.get("poem_type", DEFAULT_POEM_TYPE),
                tone=extracted_context.get("tone", DEFAULT_TONE),
                theme=extracted_context.get("theme", "general"),
                rhyme_scheme=extracted_context.get("rhyme_scheme", "free_verse"),
                original_query=original_query
            )
            
            # Add feedback if this is a regeneration
            if attempt > 0 and feedback_section:
                user_prompt += feedback_section
                logger.info(f"   📝 Added feedback for attempt {attempt}")
            
            # Create messages for Gemini
            messages = [
                SystemMessage(content=POEM_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            # Increase creativity for retries
            current_temp = min(TEMPERATURE + (attempt * 0.05), 1.0)
            
            logger.info(f"   🎲 Temperature: {current_temp:.2f}")
            logger.info(f"   📏 Max tokens: {MAX_TOKENS}")
            
            # Create LLM instance
            current_llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=current_temp,
                max_output_tokens=MAX_TOKENS
            )
            
            # Generate poem
            logger.info("   🤖 Calling Gemini API...")
            response = current_llm.invoke(messages)
            
            # Extract the poem text
            generated_text = self._extract_content(response)
            
            # Handle truncation (if poem was cut off)
            finish_reason = self._get_finish_reason(response)
            if finish_reason == 'MAX_TOKENS':
                logger.warning("⚠️ Response truncated due to MAX_TOKENS")
                
                max_tokens_retry_count = state.get("max_tokens_retry_count", 0)
                
                # If poem seems incomplete and we haven't retried yet
                if not self._is_poem_complete(generated_text) and max_tokens_retry_count < 1:
                    logger.info("   🔄 Retrying with 2x token limit...")
                    
                    state["max_tokens_retry_count"] = max_tokens_retry_count + 1
                    
                    # Retry with double tokens
                    retry_llm = ChatGoogleGenerativeAI(
                        model=GEMINI_MODEL,
                        google_api_key=GEMINI_API_KEY,
                        temperature=current_temp,
                        max_output_tokens=MAX_TOKENS * 2
                    )
                    
                    retry_response = retry_llm.invoke(messages)
                    generated_text = self._extract_content(retry_response)
                    
                    logger.info("   ✅ Retry successful")
            
            # Store poem in state
            state["generated_poem"] = generated_text
            
            logger.info(f"✅ Poem generated:")
            logger.info(f"   Length: {len(generated_text)} chars")
            logger.info(f"   Words: {len(generated_text.split())} words")
            logger.info(f"   Preview: {generated_text[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ Poem generation failed: {e}", exc_info=True)
            state["generated_poem"] = f"Error generating poem: {str(e)}"
        
        return state
    
    def _evaluate_poem_node(self, state: PoemAgentState) -> PoemAgentState:
        """
        NODE 3: Evaluate Poem (OPTIONAL)
        
        PURPOSE: Check if the poem is good quality
        
        WHAT IT DOES:
        1. If evaluator is disabled → skip and return
        2. If enabled → call PoemEvaluator tool
        3. Get a score (0-10)
        4. Store in state["evaluation"]
        
        WHY OPTIONAL?
        - Costs extra API calls (more $$$)
        - Adds latency (slower)
        - But improves quality!
        
        Enable in production, disable in development.
        """
        logger.info("📊 NODE 3: Evaluating poem...")
        
        # Check if evaluator is enabled
        if not USE_EVALUATOR:
            logger.info("⚠️ Evaluator disabled (save API costs)")
            
            # Skip evaluation
            state["evaluation"] = {
                "score": 10.0,
                "passed": True,
                "feedback": "Evaluator disabled"
            }
            state["final_poem"] = state.get("generated_poem", "")
            
            return state
        
        # Evaluator is enabled, let's check quality
        try:
            # Remove feedback from prompt before evaluating
            original_request = state["prompt"].split("=== FEEDBACK")[0].strip()
            
            # Call evaluator
            evaluation = self.poem_evaluator.evaluate(
                original_request,
                state.get("generated_poem", "")
            )
            
            # Store in state
            state["evaluation"] = evaluation
            
            logger.info(f"✅ Evaluation complete:")
            logger.info(f"   Score: {evaluation.get('score', 0):.1f}/10")
            logger.info(f"   Passed: {evaluation.get('passed', False)}")
            
            if not evaluation.get('passed'):
                logger.warning(f"   Issues: {evaluation.get('critical_errors', [])}")
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            
            # Fallback: assume it passed
            state["evaluation"] = {
                "score": 10.0,
                "passed": True,
                "feedback": f"Evaluation error: {e}"
            }
        
        # Set final poem
        state["final_poem"] = state.get("generated_poem", "")
        
        return state
    
    def _regenerate_if_needed_node(self, state: PoemAgentState) -> PoemAgentState:
        """
        NODE 4: Prepare for Regeneration
        
        PURPOSE: Add feedback to prompt so next attempt is better
        
        WHAT IT DOES:
        1. Get evaluation feedback
        2. Add it to the prompt
        3. Increment attempt counter
        4. Next iteration will see this feedback and improve
        """
        logger.info("🔄 NODE 4: Preparing regeneration...")
        
        evaluation = state.get("evaluation", {})
        attempt = state.get("attempt", 0)
        
        # Check if we've hit max retries
        if attempt >= EVALUATOR_MAX_RETRIES:
            logger.warning(f"⚠️ Max retries ({EVALUATOR_MAX_RETRIES}) reached")
            return state
        
        # Increment attempt
        state["attempt"] = attempt + 1
        logger.info(f"   Attempt: {state['attempt']}/{EVALUATOR_MAX_RETRIES}")
        
        # Extract feedback
        feedback = evaluation.get("feedback", "")
        critical_errors = evaluation.get("critical_errors", [])
        suggestions = evaluation.get("suggestions", [])
        score = evaluation.get("score", 0)
        
        # Build feedback text
        if feedback or critical_errors or suggestions:
            feedback_text = f"\n\n=== FEEDBACK FROM PREVIOUS ATTEMPT (Score: {score:.1f}/10) ===\n"
            
            if critical_errors:
                feedback_text += "CRITICAL ERRORS TO FIX:\n"
                for i, error in enumerate(critical_errors, 1):
                    feedback_text += f"{i}. {error}\n"
                feedback_text += "\n"
            
            if suggestions:
                feedback_text += "SUGGESTIONS FOR IMPROVEMENT:\n"
                for i, suggestion in enumerate(suggestions, 1):
                    feedback_text += f"{i}. {suggestion}\n"
                feedback_text += "\n"
            
            if feedback:
                feedback_text += f"DETAILED FEEDBACK:\n{feedback}\n"
            
            # Remove old feedback (if any)
            original_prompt = state.get("prompt", "")
            if "=== FEEDBACK" in original_prompt:
                original_prompt = original_prompt.split("=== FEEDBACK")[0].strip()
            
            # Add new feedback
            state["prompt"] = original_prompt + feedback_text
            
            logger.info(f"   ✅ Added feedback ({len(feedback_text)} chars)")
        else:
            logger.warning("   ⚠️ No feedback available")
        
        return state
    
    # ============================================
    # STEP 5: DECISION FUNCTION
    # ============================================
    
    def _should_regenerate(self, state: PoemAgentState) -> str:
        """
        DECISION FUNCTION: Should we regenerate or end?
        
        This is called after evaluation to decide what to do next.
        
        LOGIC:
        1. If evaluator is disabled → always end
        2. If score is low AND attempts < max → regenerate
        3. Otherwise → end
        
        RETURNS:
        - "regenerate" → Go to regenerate_if_needed node
        - "end" → Stop workflow and return result
        """
        logger.info("🤔 DECISION: Should we regenerate?")
        
        # If evaluator is disabled, always end
        if not USE_EVALUATOR:
            logger.info("   → END (evaluator disabled)")
            return "end"
        
        # Get evaluation data
        evaluation = state.get("evaluation", {})
        attempt = state.get("attempt", 0)
        passed = evaluation.get("passed", False)
        score = evaluation.get("score", 0)
        
        # Decision logic
        if not passed and attempt < EVALUATOR_MAX_RETRIES:
            logger.info(f"   → REGENERATE (score {score:.1f} < {EVALUATOR_THRESHOLD}, attempt {attempt+1}/{EVALUATOR_MAX_RETRIES})")
            return "regenerate"
        elif attempt >= EVALUATOR_MAX_RETRIES:
            logger.info(f"   → END (max retries reached, final score: {score:.1f})")
            return "end"
        else:
            logger.info(f"   → END (passed with score {score:.1f})")
            return "end"
    
    # ============================================
    # STEP 6: HELPER METHODS
    # ============================================
    
    def _extract_content(self, response) -> str:
        """Extract text from LLM response."""
        if hasattr(response, 'content'):
            return str(response.content).strip()
        if hasattr(response, 'text'):
            return str(response.text).strip()
        return str(response).strip()
    
    def _get_finish_reason(self, response) -> str:
        """Get finish reason from response metadata."""
        if hasattr(response, 'response_metadata'):
            return response.response_metadata.get('finish_reason', '')
        return ''
    
    def _is_poem_complete(self, text: str) -> bool:
        """Check if poem appears complete (not truncated)."""
        if not text or len(text.strip()) < 50:
            return False
        
        # Check if ends with "..." (truncated)
        ends_properly = not text.rstrip().endswith('...')
        
        # Check if has reasonable length
        has_length = len(text.split()) > 20
        
        return ends_properly and has_length
    
    # ============================================
    # STEP 7: PUBLIC API
    # ============================================
    
    def generate(
        self,
        prompt: str,
        enhanced_query: Optional[Dict[str, Any]] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        PUBLIC API: Generate a poem.
        
        This is the main function that main.py calls!
        
        PARAMETERS:
        - prompt: User's request (e.g., "Write a sad poem about rain")
        - enhanced_query: Extra info from router (optional)
        - max_length: Token limit (optional, uses config default)
        - temperature: Creativity (optional, uses config default)
        
        RETURNS:
        - The generated poem (string)
        
        HOW IT WORKS:
        1. Creates initial state
        2. Runs the LangGraph workflow
        3. Extracts final poem from state
        4. Returns poem
        """
        logger.info("=" * 60)
        logger.info("📝 PoemExpertAgent.generate() called")
        logger.info(f"   Prompt: {prompt[:100]}...")
        logger.info("=" * 60)
        
        # Handle None
        if enhanced_query is None:
            enhanced_query = {}
        
        # Create initial state
        initial_state: PoemAgentState = {
            "prompt": prompt,
            "enhanced_query": enhanced_query,
            "extracted_context": None,
            "generated_poem": None,
            "evaluation": None,
            "attempt": 0,
            "final_poem": "",
            "max_length": max_length or MAX_TOKENS,
            "temperature": temperature or TEMPERATURE,
            "max_tokens_retry_count": 0
        }
        
        logger.info("Initial state created")
        
        # Run the workflow
        try:
            logger.info("🚀 Invoking LangGraph workflow...")
            logger.info("")
            
            # This runs the entire workflow
            final_state = self.workflow.invoke(initial_state)
            
            logger.info("")
            logger.info("=" * 60)
            logger.info("✅ Workflow completed successfully")
            
            # Extract result
            result = final_state.get("final_poem", "")
            
            if not result:
                logger.warning("⚠️ No poem in final_poem, using fallback")
                result = final_state.get("generated_poem", "Error: No poem generated")
            
            logger.info(f"📤 Returning poem: {len(result)} chars")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ Workflow failed: {e}", exc_info=True)
            logger.error("=" * 60)
            return f"Error generating poem: {str(e)}"