"""
Story Expert LangGraph Agent
LangGraph-based agent for generating creative stories with multi-step workflow.
"""
import logging
from typing import Dict, Any, Optional, TypedDict, Tuple
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import (
    GEMINI_API_KEY, GEMINI_MODEL, MAX_TOKENS, TEMPERATURE,
    USE_EVALUATOR, EVALUATOR_THRESHOLD, EVALUATOR_MAX_RETRIES,
    DEFAULT_GENRE, DEFAULT_TONE,
    USE_CONTEXT_EXTRACTOR,
    USE_STORY_PLANNER,
    USE_CHARACTER_GENERATOR
)

from .prompts import STORY_SYSTEM_PROMPT, STORY_USER_PROMPT_TEMPLATE
from .tools import (
    ContextExtractor,
    StoryPlanner,
    StoryWriter,
    StoryEvaluator,
    CharacterGenerator
)

logger = logging.getLogger(__name__)


class StoryAgentState(TypedDict):
    """State schema for LangGraph story agent."""
    prompt: str
    enhanced_query: Optional[Dict[str, Any]]
    extracted_context: Optional[Dict[str, Any]]
    story_plan: Optional[Dict[str, Any]]
    character_profile: Optional[str]
    generated_story: Optional[str]
    evaluation: Optional[Dict[str, Any]]
    attempt: int
    final_story: str
    max_length: Optional[int]
    temperature: Optional[float]
    max_tokens_retry_count: int


class StoryExpertAgent:
    """LangGraph-based story expert agent with multi-step workflow."""
    
    def __init__(self):
        """Initialize the story expert agent with all tools and LangGraph workflow."""
        logger.info("🚀 Initializing Story Expert Agent...")
        
        # Initialize tools
        self.context_extractor = ContextExtractor(api_key=GEMINI_API_KEY)
        self.story_planner = StoryPlanner(api_key=GEMINI_API_KEY)
        self.story_writer = StoryWriter()
        self.story_evaluator = StoryEvaluator(
            api_key=GEMINI_API_KEY,
            threshold=EVALUATOR_THRESHOLD,
            max_retries=EVALUATOR_MAX_RETRIES
        )
        self.character_generator = CharacterGenerator(api_key=GEMINI_API_KEY)
        
        # Initialize Gemini LLM
        if GEMINI_API_KEY:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GEMINI_API_KEY,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS
                )
                logger.info(f"✅ LLM initialized: {GEMINI_MODEL}")
            except Exception as e:
                logger.error(f"❌ LLM initialization failed: {e}")
                self.llm = None
        else:
            logger.warning("⚠️ No GEMINI_API_KEY. LLM disabled.")
            self.llm = None
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        logger.info("✅ Story Expert Agent initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(StoryAgentState)
        
        # Add nodes
        workflow.add_node("extract_context", self._extract_context_node)
        workflow.add_node("plan_story", self._plan_story_node)
        workflow.add_node("generate_story", self._generate_story_node)
        workflow.add_node("evaluate_story", self._evaluate_story_node)
        workflow.add_node("regenerate_if_needed", self._regenerate_if_needed_node)
        
        # Define workflow edges
        workflow.set_entry_point("extract_context")
        workflow.add_edge("extract_context", "plan_story")
        workflow.add_edge("plan_story", "generate_story")
        workflow.add_edge("generate_story", "evaluate_story")
        
        # Conditional edge: evaluate -> regenerate or end
        workflow.add_conditional_edges(
            "evaluate_story",
            self._should_regenerate,
            {
                "regenerate": "regenerate_if_needed",
                "end": END
            }
        )
        
        # Regenerate loops back to generate_story
        workflow.add_edge("regenerate_if_needed", "generate_story")
        
        return workflow.compile()
    
    def _extract_context_node(self, state: StoryAgentState) -> StoryAgentState:
        """Extract story context from prompt."""
        logger.info("🔍 Extracting story context...")
        
        # Check if disabled
        if not USE_CONTEXT_EXTRACTOR:
            logger.info("⚠️ Context extractor DISABLED (saving API calls)")
            state["extracted_context"] = {
                "story_type": "short_story",
                "genre": DEFAULT_GENRE,
                "tone": DEFAULT_TONE,
                "themes": [],
                "setting": {},
                "characters": {},
                "plot_elements": {},
                "style_preferences": {},
                "length_target": "medium",
                "special_requirements": []
            }
            return state
        
        # Extract context using API
        try:
            extracted_context = self.context_extractor.extract(
                state["prompt"],
                state.get("enhanced_query")
            )
            state["extracted_context"] = extracted_context
            logger.info(f"✅ Context extracted: Genre={extracted_context.get('genre')}, Tone={extracted_context.get('tone')}")
        except Exception as e:
            logger.error(f"❌ Context extraction failed: {e}")
            state["extracted_context"] = {
                "story_type": "short_story",
                "genre": DEFAULT_GENRE,
                "tone": DEFAULT_TONE,
                "themes": [],
                "setting": {},
                "characters": {},
                "plot_elements": {},
                "style_preferences": {},
                "length_target": "medium",
                "special_requirements": []
            }
        
        return state
    
    def _plan_story_node(self, state: StoryAgentState) -> StoryAgentState:
        """Generate story plan."""
        logger.info("📋 Planning story...")
        
        if not USE_STORY_PLANNER:
            logger.info("⚠️ Story planner DISABLED (saving API calls)")
            state["story_plan"] = {
                "story_structure": {
                    "opening_hook": "Begin the story with an engaging hook",
                    "rising_action": ["Develop the plot", "Build tension", "Introduce conflict"],
                    "climax": "Reach the story's climax",
                    "falling_action": "Resolve the main conflict",
                    "conclusion": "End with a satisfying resolution"
                }
            }
            state["character_profile"] = "A compelling protagonist"
            return state
        
        try:
            extracted_context = state.get("extracted_context", {})
            
            # Generate character profile if needed
            genre = extracted_context.get("genre", DEFAULT_GENRE)
            character_profile = self.character_generator.generate("protagonist", genre)
            
            # Generate story plan
            story_plan = self.story_planner.plan(extracted_context)
            
            state["character_profile"] = character_profile
            state["story_plan"] = story_plan
            logger.info("✅ Story planned successfully")
        except Exception as e:
            logger.error(f"❌ Story planning failed: {e}")
            state["story_plan"] = {
                "story_structure": {
                    "opening_hook": "Begin the story",
                    "rising_action": ["Develop plot"],
                    "climax": "Reach climax",
                    "falling_action": "Resolve conflicts",
                    "conclusion": "End the story"
                }
            }
            state["character_profile"] = "A protagonist on a journey"
        
        return state
    
    def _generate_story_node(self, state: StoryAgentState) -> StoryAgentState:
        """Generate story using Gemini."""
        logger.info("✍️ Generating story...")
        
        if not self.llm:
            logger.warning("⚠️ LLM not available, using fallback")
            state["generated_story"] = "Error: LLM not initialized"
            return state
        
        try:
            # Build context
            extracted_context = state.get("extracted_context", {})
            story_plan = state.get("story_plan", {})
            character_profile = state.get("character_profile", "")
            attempt = state.get("attempt", 0)
            
            # Extract original query (before feedback)
            current_prompt = state.get("prompt", "")
            original_query = current_prompt
            feedback_section = ""
            if "=== CRITICAL FEEDBACK" in current_prompt:
                parts = current_prompt.split("=== CRITICAL FEEDBACK", 1)
                original_query = parts[0].strip()
                feedback_section = "\n\n=== CRITICAL FEEDBACK" + parts[1] if len(parts) > 1 else ""
            
            # Build key elements context
            key_elements = f"""CHARACTER PROFILE:
{character_profile}

STORY STRUCTURE:
{story_plan.get('story_structure', {})}

THEMES: {', '.join(extracted_context.get('themes', []))}"""
            
            # Build user prompt
            user_prompt = STORY_USER_PROMPT_TEMPLATE.format(
                enhanced_instruction=state.get("enhanced_query", {}).get("enhanced_instruction", original_query),
                genre=extracted_context.get("genre", DEFAULT_GENRE),
                tone=extracted_context.get("tone", DEFAULT_TONE),
                key_elements=key_elements,
                length_preference=extracted_context.get("length_target", "medium"),
                original_query=original_query,
                additional_context=feedback_section if attempt > 0 else ""
            )
            
            # Add feedback if regenerating
            if attempt > 0 and feedback_section:
                logger.info(f"   ✅ Added feedback for regeneration attempt {attempt}")
            
            # Generate story
            messages = [
                SystemMessage(content=STORY_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            # Use persistent counter from state
            max_tokens_retry_count = state.get("max_tokens_retry_count", 0)
            max_retries = 1
            
            # Use adjusted temperature for regenerations (decrease not increase)
            current_temp = max(TEMPERATURE - (attempt * 0.05), 0.5)
            
            current_llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=current_temp,
                max_output_tokens=MAX_TOKENS
            )
            
            response = current_llm.invoke(messages)
            
            # Extract content
            generated_text = self._extract_content(response)
            
            # Check for truncation
            finish_reason = self._get_finish_reason(response)
            if finish_reason == 'MAX_TOKENS' and max_tokens_retry_count < max_retries:
                if not self._is_story_complete(generated_text):
                    logger.warning(f"⚠️ Story truncated, retrying with 2x tokens...")
                    state["max_tokens_retry_count"] = max_tokens_retry_count + 1
                    
                    retry_llm = ChatGoogleGenerativeAI(
                        model=GEMINI_MODEL,
                        google_api_key=GEMINI_API_KEY,
                        temperature=current_temp,
                        max_output_tokens=MAX_TOKENS * 2
                    )
                    retry_response = retry_llm.invoke(messages)
                    generated_text = self._extract_content(retry_response)
                    logger.info("✅ Retry successful")
            
            # Format story
            generated_text = self.story_writer.format_story(generated_text)
            
            state["generated_story"] = generated_text
            logger.info(f"✅ Story generated: {len(generated_text)} chars, {len(generated_text.split())} words")
            
        except Exception as e:
            logger.error(f"❌ Story generation failed: {e}", exc_info=True)
            state["generated_story"] = f"Error generating story: {str(e)}"
        
        return state
    
    def _evaluate_story_node(self, state: StoryAgentState) -> StoryAgentState:
        """Evaluate story quality."""
        logger.info("📊 Evaluating story...")
        
        if not USE_EVALUATOR:
            logger.info("⚠️ Evaluator disabled, skipping evaluation")
            state["evaluation"] = {"score": 10.0, "passed": True, "feedback": "Evaluator disabled"}
            state["final_story"] = state.get("generated_story", "")
            return state
        
        try:
            evaluation = self.story_evaluator.evaluate(
                state["prompt"].split("=== CRITICAL FEEDBACK")[0].strip(),
                state.get("generated_story", "")
            )
            state["evaluation"] = evaluation
            logger.info(f"📊 Evaluation score: {evaluation.get('score', 0):.1f}, Passed: {evaluation.get('passed', False)}")
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            state["evaluation"] = {"score": 10.0, "passed": True, "feedback": f"Evaluation error: {e}"}
        
        state["final_story"] = state.get("generated_story", "")
        return state
    
    def _regenerate_if_needed_node(self, state: StoryAgentState) -> StoryAgentState:
        """Prepare for regeneration with feedback."""
        logger.info("🔄 Preparing regeneration with feedback...")
        
        evaluation = state.get("evaluation", {})
        attempt = state.get("attempt", 0)
        
        if attempt >= EVALUATOR_MAX_RETRIES:
            logger.warning(f"⚠️ Max retries reached")
            return state
        
        state["attempt"] = attempt + 1
        
        # Add feedback to prompt
        feedback = evaluation.get("feedback", "")
        critical_errors = evaluation.get("critical_errors", [])
        suggestions = evaluation.get("suggestions", [])
        score = evaluation.get("score", 0)
        
        if feedback or critical_errors or suggestions:
            feedback_text = f"\n\n=== CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT (Score: {score:.1f}/10) ===\n"
            if critical_errors:
                feedback_text += "CRITICAL ERRORS TO FIX:\n"
                for i, error in enumerate(critical_errors, 1):
                    feedback_text += f"{i}. {error}\n"
                feedback_text += "\n"
            if suggestions:
                feedback_text += "IMPROVEMENTS NEEDED:\n"
                for i, suggestion in enumerate(suggestions, 1):
                    feedback_text += f"{i}. {suggestion}\n"
                feedback_text += "\n"
            if feedback:
                feedback_text += f"DETAILED FEEDBACK:\n{feedback}\n\n"
            
            # Remove previous feedback
            original_prompt = state.get("prompt", "")
            if "=== CRITICAL FEEDBACK" in original_prompt:
                original_prompt = original_prompt.split("=== CRITICAL FEEDBACK")[0].strip()
            
            state["prompt"] = original_prompt + feedback_text
            logger.info(f"   ✅ Added feedback for attempt {state['attempt']}")
        
        return state
    
    def _should_regenerate(self, state: StoryAgentState) -> str:
        """Determine if story should be regenerated."""
        if not USE_EVALUATOR:
            return "end"
        
        evaluation = state.get("evaluation", {})
        attempt = state.get("attempt", 0)
        passed = evaluation.get("passed", False)
        
        if not passed and attempt < EVALUATOR_MAX_RETRIES:
            logger.info(f"🔄 Will regenerate (attempt {attempt + 1}/{EVALUATOR_MAX_RETRIES})")
            return "regenerate"
        
        logger.info(f"✅ Workflow complete (score: {evaluation.get('score', 0):.1f})")
        return "end"
    
    def _extract_content(self, response) -> str:
        """Extract text from LLM response."""
        if hasattr(response, 'content'):
            return str(response.content).strip()
        if hasattr(response, 'text'):
            return str(response.text).strip()
        return str(response).strip()
    
    def _get_finish_reason(self, response) -> str:
        """Extract finish reason from response."""
        if hasattr(response, 'response_metadata'):
            return response.response_metadata.get('finish_reason', '')
        return ''
    
    def _is_story_complete(self, text: str) -> bool:
        """Check if story appears complete."""
        if not text or len(text.strip()) < 200:
            return False
        
        # Check for proper ending
        ends_properly = not text.rstrip().endswith('...')
        has_length = len(text.split()) > 100
        
        return ends_properly and has_length
    
    def generate(
        self,
        prompt: str,
        enhanced_query: Optional[Dict[str, Any]] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate story using the LangGraph workflow.
        
        Args:
            prompt: User query
            enhanced_query: Enhanced query from router (optional)
            max_length: Maximum generation length (optional)
            temperature: Sampling temperature (optional)
            
        Returns:
            Generated story text
        """
        logger.info(f"📖 StoryExpertAgent.generate() called")
        logger.info(f"   Prompt: {prompt[:100]}...")
        
        # Convert None to empty dict
        if enhanced_query is None:
            enhanced_query = {}
        
        # Initialize state
        initial_state: StoryAgentState = {
            "prompt": prompt,
            "enhanced_query": enhanced_query,
            "extracted_context": None,
            "story_plan": None,
            "character_profile": None,
            "generated_story": None,
            "evaluation": None,
            "attempt": 0,
            "final_story": "",
            "max_length": max_length or MAX_TOKENS,
            "temperature": temperature or TEMPERATURE,
            "max_tokens_retry_count": 0
        }
        
        # Run workflow
        try:
            logger.info("🚀 Invoking LangGraph workflow...")
            final_state = self.workflow.invoke(initial_state)
            
            result = final_state.get("final_story", "")
            
            if not result:
                logger.warning("⚠️ No story generated, using fallback")
                result = "Error: Story generation completed but returned no text."
            
            logger.info(f"✅ Story generated: {len(result)} chars, {len(result.split())} words")
            return result
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}", exc_info=True)
            return f"Error generating story: {str(e)}"