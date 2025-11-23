"""
Email Expert LangGraph Agent
LangGraph-based agent for generating professional emails with multi-step workflow.
"""
import os
import logging
from typing import Dict, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import (
    GEMINI_API_KEY, GEMINI_MODEL, MAX_TOKENS, TEMPERATURE,
    USE_EVALUATOR, EVALUATOR_THRESHOLD, EVALUATOR_MAX_RETRIES
)
from .prompts import EMAIL_SYSTEM_PROMPT, EMAIL_USER_PROMPT_TEMPLATE
from .tools import (
    ContextExtractor,
    TemplateGenerator,
    ToneTransformer,
    EmailEvaluator
)

logger = logging.getLogger(__name__)


class EmailAgentState(TypedDict):
    """State schema for LangGraph email agent."""
    prompt: str
    enhanced_query: Optional[Dict[str, Any]]
    extracted_context: Optional[Dict[str, Any]]
    email_template: Optional[Dict[str, Any]]
    tone_adjusted_content: Optional[str]
    generated_email: Optional[str]
    parsed_email: Optional[str]
    evaluation: Optional[Dict[str, Any]]
    attempt: int
    final_email: str
    max_length: Optional[int]
    temperature: Optional[float]


class EmailExpertAgent:
    """LangGraph-based email expert agent with multi-step workflow."""
    
    def __init__(self):
        """Initialize the email expert agent with all tools and LangGraph workflow."""
        logger.info("🚀 Initializing Email Expert Agent...")
        
        # Initialize tools
        self.context_extractor = ContextExtractor(api_key=GEMINI_API_KEY)
        self.template_generator = TemplateGenerator(api_key=GEMINI_API_KEY)
        self.tone_transformer = ToneTransformer(api_key=GEMINI_API_KEY)
        self.email_evaluator = EmailEvaluator(
            api_key=GEMINI_API_KEY,
            threshold=EVALUATOR_THRESHOLD,
            max_retries=EVALUATOR_MAX_RETRIES
        )
        
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
        logger.info("✅ Email Expert Agent initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(EmailAgentState)
        
        # Add nodes
        workflow.add_node("extract_context", self._extract_context_node)
        workflow.add_node("generate_template", self._generate_template_node)
        workflow.add_node("transform_tone", self._transform_tone_node)
        workflow.add_node("generate_email", self._generate_email_node)
        workflow.add_node("evaluate_email", self._evaluate_email_node)
        workflow.add_node("regenerate_if_needed", self._regenerate_if_needed_node)
        
        # Define workflow edges
        workflow.set_entry_point("extract_context")
        workflow.add_edge("extract_context", "generate_template")
        workflow.add_edge("generate_template", "transform_tone")
        workflow.add_edge("transform_tone", "generate_email")
        workflow.add_edge("generate_email", "evaluate_email")
        
        # Conditional edge: evaluate -> regenerate or end
        workflow.add_conditional_edges(
            "evaluate_email",
            self._should_regenerate,
            {
                "regenerate": "regenerate_if_needed",
                "end": END
            }
        )
        
        # Regenerate loops back to generate_email
        workflow.add_edge("regenerate_if_needed", "generate_email")
        
        return workflow.compile()
    
    def _extract_context_node(self, state: EmailAgentState) -> EmailAgentState:
        """Extract context from prompt."""
        logger.info("🔍 Extracting context...")
        try:
            extracted_context = self.context_extractor.extract(
                state["prompt"],
                state.get("enhanced_query")
            )
            state["extracted_context"] = extracted_context
            logger.info(f"✅ Context extracted: intent={extracted_context.get('intent')}")
        except Exception as e:
            logger.error(f"❌ Context extraction failed: {e}")
            # Fallback: create minimal context
            state["extracted_context"] = {
                "extracted_context": state["prompt"],
                "intent": "general",
                "key_entities": {},
                "relationships": [],
                "email_type": "general",
                "urgency": "normal",
                "formality_level": "professional"
            }
        return state
    
    def _generate_template_node(self, state: EmailAgentState) -> EmailAgentState:
        """Generate email template."""
        logger.info("📝 Generating template...")
        try:
            email_template = self.template_generator.generate(
                state.get("extracted_context", {}),
                state.get("enhanced_query")
            )
            state["email_template"] = email_template
            logger.info(f"✅ Template generated: {len(email_template.get('email_template', ''))} chars")
        except Exception as e:
            logger.error(f"❌ Template generation failed: {e}")
            # Fallback: create minimal template
            state["email_template"] = {
                "email_template": "Subject: Professional Correspondence\n\nDear Recipient,\n\n[Content]\n\nBest regards,\n[Your Name]",
                "structure": {"has_subject": True, "has_greeting": True, "has_body": True, "has_closing": True},
                "sections": {"subject": "Professional Correspondence", "greeting": "Dear Recipient,", "body": "[Content]", "closing": "Best regards,\n[Your Name]"}
            }
        return state
    
    def _transform_tone_node(self, state: EmailAgentState) -> EmailAgentState:
        """Transform email tone."""
        logger.info("🎨 Transforming tone...")
        try:
            template_content = state.get("email_template", {}).get("email_template", "")
            tone_result = self.tone_transformer.transform(
                template_content,
                state.get("extracted_context", {}),
                state.get("enhanced_query")
            )
            state["tone_adjusted_content"] = tone_result.get("tone_adjusted_email", template_content)
            logger.info(f"✅ Tone transformed: {tone_result.get('tone_analysis', 'N/A')}")
        except Exception as e:
            logger.error(f"❌ Tone transformation failed: {e}")
            # Fallback: use template as-is
            state["tone_adjusted_content"] = state.get("email_template", {}).get("email_template", "")
        return state
    
    def _generate_email_node(self, state: EmailAgentState) -> EmailAgentState:
        """Generate email using Gemini."""
        logger.info("✍️ Generating email...")
        
        if not self.llm:
            logger.warning("⚠️ LLM not available, using template")
            state["generated_email"] = state.get("tone_adjusted_content", "")
            return state
        
        try:
            # Build prompt from context
            extracted_context = state.get("extracted_context", {})
            enhanced_query = state.get("enhanced_query", {})
            tone_adjusted = state.get("tone_adjusted_content", "")
            
            # Create enhanced instruction
            enhanced_instruction = enhanced_query.get("enhanced_instruction", state["prompt"])
            email_type = enhanced_query.get("email_type", extracted_context.get("email_type", "general"))
            tone = enhanced_query.get("tone", extracted_context.get("formality_level", "professional"))
            recipient_type = enhanced_query.get("recipient_type", extracted_context.get("key_entities", {}).get("recipient", "general"))
            key_points = enhanced_query.get("key_points", [])
            special_requirements = enhanced_query.get("special_requirements", [])
            
            # Build user prompt
            user_prompt = EMAIL_USER_PROMPT_TEMPLATE.format(
                enhanced_instruction=enhanced_instruction,
                email_type=email_type,
                tone=tone,
                recipient_type=recipient_type,
                key_points=", ".join(key_points) if key_points else "N/A",
                special_requirements=", ".join(special_requirements) if special_requirements else "N/A",
                original_query=state["prompt"]
            )
            
            # Add template context if available
            if tone_adjusted:
                user_prompt += f"\n\nUse this template structure as a guide:\n{tone_adjusted}"
            
            # Generate email
            messages = [
                SystemMessage(content=EMAIL_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            generated_text = response.content if hasattr(response, 'content') else str(response)
            
            state["generated_email"] = generated_text
            logger.info(f"✅ Email generated: {len(generated_text)} chars")
            
        except Exception as e:
            logger.error(f"❌ Email generation failed: {e}")
            # Fallback: use tone-adjusted content
            state["generated_email"] = state.get("tone_adjusted_content", state["prompt"])
        
        return state
    
    def _evaluate_email_node(self, state: EmailAgentState) -> EmailAgentState:
        """Evaluate email quality."""
        logger.info("📊 Evaluating email...")
        
        if not USE_EVALUATOR:
            logger.info("⚠️ Evaluator disabled, skipping evaluation")
            state["evaluation"] = {"score": 10.0, "passed": True, "feedback": "Evaluator disabled"}
            state["final_email"] = state.get("generated_email", "")
            return state
        
        try:
            evaluation = self.email_evaluator.evaluate(
                state["prompt"],
                state.get("generated_email", "")
            )
            state["evaluation"] = evaluation
            logger.info(f"📊 Evaluation score: {evaluation.get('score', 0):.1f}, Passed: {evaluation.get('passed', False)}")
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            # Fallback: assume passed
            state["evaluation"] = {"score": 10.0, "passed": True, "feedback": f"Evaluation error: {e}"}
        
        # Set final email
        state["final_email"] = state.get("generated_email", "")
        
        return state
    
    def _regenerate_if_needed_node(self, state: EmailAgentState) -> EmailAgentState:
        """Regenerate email if needed by adding feedback to prompt."""
        logger.info("🔄 Regenerating email with feedback...")
        
        evaluation = state.get("evaluation", {})
        attempt = state.get("attempt", 0)
        
        if attempt >= EVALUATOR_MAX_RETRIES:
            logger.warning(f"⚠️ Max retries ({EVALUATOR_MAX_RETRIES}) reached")
            return state
        
        # Increment attempt
        state["attempt"] = attempt + 1
        
        # Add feedback to prompt for regeneration
        feedback = evaluation.get("feedback", "")
        critical_errors = evaluation.get("critical_errors", [])
        
        if feedback or critical_errors:
            feedback_text = f"\n\nIMPORTANT FEEDBACK FROM PREVIOUS ATTEMPT (Score: {evaluation.get('score', 0):.1f}):\n"
            if critical_errors:
                feedback_text += f"Critical errors to fix: {', '.join(critical_errors[:3])}\n"
            if feedback:
                feedback_text += f"Feedback: {feedback}\n"
            feedback_text += "Please regenerate the email addressing these issues."
            
            # Update prompt with feedback
            state["prompt"] = state["prompt"] + feedback_text
            logger.info(f"   Added feedback: {len(feedback_text)} chars")
        
        # Slightly increase temperature for regeneration
        current_temp = state.get("temperature", TEMPERATURE)
        state["temperature"] = min(current_temp + (attempt * 0.1), 1.0)
        
        return state
    
    def _should_regenerate(self, state: EmailAgentState) -> str:
        """Determine if email should be regenerated."""
        if not USE_EVALUATOR:
            return "end"
        
        evaluation = state.get("evaluation", {})
        attempt = state.get("attempt", 0)
        
        # Regenerate if not passed and haven't exceeded max retries
        if not evaluation.get("passed", False) and attempt < EVALUATOR_MAX_RETRIES:
            logger.info(f"🔄 Will regenerate (attempt {attempt + 1}/{EVALUATOR_MAX_RETRIES})")
            return "regenerate"
        
        return "end"
    
    def generate(
        self,
        prompt: str,
        enhanced_query: Optional[Dict[str, Any]] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate email using the LangGraph workflow.
        
        Args:
            prompt: User query
            enhanced_query: Enhanced query from router (optional)
            max_length: Maximum generation length (optional, uses config default)
            temperature: Sampling temperature (optional, uses config default)
            
        Returns:
            Generated email text
        """
        logger.info(f"📧 EmailExpertAgent.generate() called")
        logger.info(f"   Prompt: {prompt[:100]}...")
        logger.debug(f"   Enhanced query provided: {enhanced_query is not None}")
        logger.debug(f"   max_length: {max_length}, temperature: {temperature}")
        if enhanced_query:
            logger.debug(f"   Enhanced query keys: {list(enhanced_query.keys())}")
        
        # Initialize state
        logger.debug("📝 Initializing workflow state...")
        initial_state: EmailAgentState = {
            "prompt": prompt,
            "enhanced_query": enhanced_query,
            "extracted_context": None,
            "email_template": None,
            "tone_adjusted_content": None,
            "generated_email": None,
            "evaluation": None,
            "attempt": 0,
            "final_email": "",
            "max_length": max_length or MAX_TOKENS,
            "temperature": temperature or TEMPERATURE
        }
        logger.debug(f"   State initialized with max_length={initial_state['max_length']}, temperature={initial_state['temperature']}")
        
        # Run workflow
        try:
            logger.info("🚀 Invoking LangGraph workflow...")
            final_state = self.workflow.invoke(initial_state)
            logger.debug("   Workflow execution completed")
            
            result = final_state.get("final_email", "")
            logger.debug(f"   Final email from state: {len(result)} chars")
            
            if not result:
                logger.warning("⚠️ No email generated in final_state, using fallback")
                result = f"Subject: Professional Correspondence\n\nDear Recipient,\n\n{prompt}\n\nBest regards,\n[Your Name]"
            
            logger.info(f"✅ Email generated successfully: {len(result)} chars")
            logger.debug(f"   Email preview: {result[:150]}...")
            return result
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}", exc_info=True)
            logger.warning("⚠️ Returning fallback email")
            # Fallback email
            fallback = f"Subject: Professional Correspondence\n\nDear Recipient,\n\n{prompt}\n\nBest regards,\n[Your Name]"
            logger.debug(f"   Fallback email: {len(fallback)} chars")
            return fallback
