"""
Email Expert LangGraph Agent
LangGraph-based agent for generating professional emails with multi-step workflow.
"""
import logging
from typing import Dict, Any, Optional, TypedDict, Tuple
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
    
    def _extract_usage_metadata(self, response) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
        """
        Extract usage metadata, response metadata, and finish reason from LLM response.
        
        Returns:
            Tuple of (response_metadata, usage_metadata, finish_reason)
        """
        response_metadata = {}
        usage_metadata = {}
        finish_reason = None
        
        # Check multiple locations for response metadata
        if hasattr(response, 'response_metadata'):
            response_metadata = response.response_metadata or {}
            finish_reason = response_metadata.get('finish_reason', '')
        
        # Check for usage_metadata in multiple possible locations
        # Location 1: response.response_metadata['usage_metadata']
        if isinstance(response_metadata, dict) and 'usage_metadata' in response_metadata:
            usage_metadata = response_metadata['usage_metadata']
            logger.debug(f"   Found usage_metadata in response.response_metadata['usage_metadata']")
        # Location 2: response.usage_metadata (direct attribute)
        elif hasattr(response, 'usage_metadata'):
            usage_metadata = response.usage_metadata or {}
            logger.debug(f"   Found usage_metadata in response.usage_metadata")
        # Location 3: Check if response_metadata itself contains token info
        elif isinstance(response_metadata, dict):
            # Sometimes token info is directly in response_metadata
            if 'input_tokens' in response_metadata or 'output_tokens' in response_metadata:
                usage_metadata = response_metadata
                logger.debug(f"   Found token info directly in response_metadata")
        
        # Log token usage if found
        if usage_metadata:
            input_tokens = usage_metadata.get('input_tokens', 0)
            output_tokens = usage_metadata.get('output_tokens', 0)
            total_tokens = usage_metadata.get('total_tokens', 0)
            if input_tokens > 0 or output_tokens > 0 or total_tokens > 0:
                logger.info(f"   📊 Token usage: input={input_tokens}, output={output_tokens}, total={total_tokens}")
                logger.debug(f"   Usage metadata: {usage_metadata}")
            else:
                logger.debug(f"   Usage metadata found but empty: {usage_metadata}")
        else:
            # Debug: log response structure to help identify where token data is
            logger.debug(f"   ⚠️ Usage metadata not found. Inspecting response structure...")
            logger.debug(f"   Response type: {type(response)}")
            logger.debug(f"   Response has response_metadata: {hasattr(response, 'response_metadata')}")
            if hasattr(response, 'response_metadata'):
                logger.debug(f"   response_metadata type: {type(response_metadata)}")
                logger.debug(f"   response_metadata keys: {list(response_metadata.keys()) if isinstance(response_metadata, dict) else 'N/A'}")
            logger.debug(f"   Response has usage_metadata: {hasattr(response, 'usage_metadata')}")
            if hasattr(response, 'usage_metadata'):
                logger.debug(f"   response.usage_metadata: {response.usage_metadata}")
        
        return response_metadata, usage_metadata, finish_reason
    
    def _extract_content_from_response(self, response) -> Optional[str]:
        """
        Extract text content from LLM response using multiple fallback methods.
        
        Returns:
            Extracted text content or None if not found
        """
        generated_text = None
        
        # Method 1: Direct content attribute (most common)
        if hasattr(response, 'content'):
            content_value = response.content
            logger.debug(f"   response.content type: {type(content_value)}, value length: {len(str(content_value)) if content_value else 0}")
            if content_value:
                generated_text = content_value
                logger.debug(f"   ✅ Extracted from response.content")
        
        # Method 2: Text attribute
        if not generated_text and hasattr(response, 'text'):
            text_value = response.text
            logger.debug(f"   response.text type: {type(text_value)}, value length: {len(str(text_value)) if text_value else 0}")
            if text_value:
                generated_text = text_value
                logger.debug(f"   ✅ Extracted from response.text")
        
        # Method 3: String conversion
        if not generated_text:
            str_response = str(response)
            logger.debug(f"   str(response) length: {len(str_response)}")
            if str_response and str_response.strip():
                generated_text = str_response
                logger.debug(f"   ✅ Extracted from str(response)")
        
        # Method 4: Check for nested message/content
        if not generated_text:
            if hasattr(response, 'message'):
                logger.debug(f"   Found response.message: {type(response.message)}")
                if hasattr(response.message, 'content'):
                    msg_content = response.message.content
                    logger.debug(f"   response.message.content type: {type(msg_content)}, length: {len(str(msg_content)) if msg_content else 0}")
                    if msg_content:
                        generated_text = msg_content
                        logger.debug(f"   ✅ Extracted from response.message.content")
        
        # Method 5: Check for any other text-like attributes (excluding 'text' already checked)
        if not generated_text:
            for attr in ['message', 'output', 'result', 'data']:
                if hasattr(response, attr):
                    attr_value = getattr(response, attr)
                    logger.debug(f"   response.{attr}: {type(attr_value)}, length: {len(str(attr_value)) if attr_value else 0}")
                    if attr_value and isinstance(attr_value, str) and attr_value.strip():
                        generated_text = attr_value
                        logger.debug(f"   ✅ Extracted from response.{attr}")
                        break
        
        # Convert to string and strip
        if generated_text:
            generated_text = str(generated_text).strip()
            logger.debug(f"   Final generated_text length: {len(generated_text)}")
        
        return generated_text
    
    def _is_email_complete(self, text: str) -> bool:
        """Check if email content appears complete."""
        if not text or len(text.strip()) < 50:
            return False
        # Check for basic email structure indicators
        has_subject = 'subject:' in text.lower()[:200] or 'subject' in text.lower()[:200]
        has_body = len(text) > 100  # Reasonable body length
        # Check if it ends abruptly (common sign of truncation)
        ends_abruptly = text.rstrip().endswith('...') or text.rstrip().endswith('[')
        return has_body and not ends_abruptly
    
    def _generate_email_node(self, state: EmailAgentState) -> EmailAgentState:
        """Generate email using Gemini."""
        logger.info("✍️ Generating email...")
        
        if not self.llm:
            logger.warning("⚠️ LLM not available, using template")
            state["generated_email"] = state.get("tone_adjusted_content", "")
            return state
        
        try:
            # Reset MAX_TOKENS retry counter for new generation attempt
            # (This is separate from the regeneration attempt counter)
            state["max_tokens_retry_count"] = 0
            
            # Build prompt from context
            extracted_context = state.get("extracted_context", {})
            enhanced_query = state.get("enhanced_query", {})
            tone_adjusted = state.get("tone_adjusted_content", "")
            attempt = state.get("attempt", 0)
            
            # Get current prompt (may include feedback if regenerating)
            current_prompt = state.get("prompt", "")
            
            # Extract original query (before feedback was added in _regenerate_if_needed_node)
            original_query = current_prompt
            feedback_section = ""
            if "=== CRITICAL FEEDBACK" in current_prompt:
                parts = current_prompt.split("=== CRITICAL FEEDBACK", 1)
                original_query = parts[0].strip()
                feedback_section = "\n\n=== CRITICAL FEEDBACK" + parts[1] if len(parts) > 1 else ""
                logger.debug(f"   Extracted original query ({len(original_query)} chars) and feedback ({len(feedback_section)} chars)")
            
            # Create enhanced instruction (use original query, not prompt with feedback)
            enhanced_instruction = enhanced_query.get("enhanced_instruction", original_query)
            email_type = enhanced_query.get("email_type", extracted_context.get("email_type", "general"))
            tone = enhanced_query.get("tone", extracted_context.get("formality_level", "professional"))
            recipient_type = enhanced_query.get("recipient_type", extracted_context.get("key_entities", {}).get("recipient", "general"))
            key_points = enhanced_query.get("key_points", [])
            special_requirements = enhanced_query.get("special_requirements", [])
            
            logger.debug(f"   Building prompt for attempt {attempt}")
            logger.debug(f"   Original query: {original_query[:100]}...")
            logger.debug(f"   Email type: {email_type}, Tone: {tone}, Recipient: {recipient_type}")
            
            # Build user prompt
            user_prompt = EMAIL_USER_PROMPT_TEMPLATE.format(
                enhanced_instruction=enhanced_instruction,
                email_type=email_type,
                tone=tone,
                recipient_type=recipient_type,
                key_points=", ".join(key_points) if key_points else "N/A",
                special_requirements=", ".join(special_requirements) if special_requirements else "N/A",
                original_query=original_query
            )
            
            # Add template context if available
            if tone_adjusted:
                user_prompt += f"\n\nUse this template structure as a guide:\n{tone_adjusted}"
            
            # Add feedback section if this is a regeneration attempt
            if attempt > 0 and feedback_section:
                user_prompt += feedback_section
                logger.info(f"   ✅ Added feedback section for regeneration attempt {attempt}")
                logger.debug(f"   Feedback section length: {len(feedback_section)} chars")
            elif attempt > 0:
                logger.warning(f"   ⚠️ Regeneration attempt {attempt} but no feedback found in prompt!")
            
            # Generate email
            logger.debug(f"   Building LLM messages (attempt {attempt})...")
            messages = [
                SystemMessage(content=EMAIL_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            logger.debug(f"   Calling LLM with prompt length: {len(user_prompt)} chars")
            logger.debug(f"   System prompt length: {len(EMAIL_SYSTEM_PROMPT)} chars")
            logger.debug(f"   Total messages: {len(messages)}")
            
            # Track retry attempts for MAX_TOKENS (reset for each generation attempt)
            max_tokens_retry_count = state.get("max_tokens_retry_count", 0)
            max_retries = 1  # Only retry once with higher limit
            
            generated_text = None
            response = None
            
            try:
                # Use current LLM instance (with MAX_TOKENS limit)
                current_llm = self.llm
                current_max_tokens = MAX_TOKENS
                
                response = current_llm.invoke(messages)
                
                # Extract response metadata using helper method
                response_metadata, usage_metadata, finish_reason = self._extract_usage_metadata(response)
                
                # Log finish reason
                if finish_reason:
                    logger.debug(f"   Finish reason: {finish_reason}")
                    if finish_reason == 'MAX_TOKENS':
                        logger.warning(f"   ⚠️ Response truncated due to MAX_TOKENS limit ({current_max_tokens})")
                        logger.warning(f"   Output tokens used: {usage_metadata.get('output_tokens', 'unknown')}")
                
                # Check for safety filters or blocked content
                if isinstance(response_metadata, dict):
                    if 'safety_ratings' in response_metadata:
                        logger.warning(f"   ⚠️ Safety ratings found: {response_metadata['safety_ratings']}")
                    if 'blocked' in response_metadata:
                        logger.warning(f"   ⚠️ Content blocked: {response_metadata['blocked']}")
                    if finish_reason in ['SAFETY', 'RECITATION', 'OTHER']:
                        logger.error(f"   ❌ Response blocked by safety filter! Finish reason: {finish_reason}")
                        logger.error(f"   Metadata: {response_metadata}")
                
                # Extract content using helper method
                generated_text = self._extract_content_from_response(response)
                
                # Handle MAX_TOKENS truncation
                if finish_reason == 'MAX_TOKENS':
                    is_complete = self._is_email_complete(generated_text) if generated_text else False
                    
                    if not is_complete and max_tokens_retry_count < max_retries:
                        logger.warning(f"   ⚠️ Content appears incomplete after MAX_TOKENS truncation")
                        logger.warning(f"   Generated text length: {len(generated_text) if generated_text else 0} chars")
                        logger.info(f"   🔄 Retrying with 2x token limit ({MAX_TOKENS * 2})...")
                        
                        # Update retry count in state
                        state["max_tokens_retry_count"] = max_tokens_retry_count + 1
                        max_tokens_retry_count = state["max_tokens_retry_count"]
                        
                        # Retry with higher limit
                        retry_llm = ChatGoogleGenerativeAI(
                            model=GEMINI_MODEL,
                            google_api_key=GEMINI_API_KEY,
                            temperature=TEMPERATURE,
                            max_output_tokens=MAX_TOKENS * 2
                        )
                        logger.debug(f"   Retrying LLM invocation with max_output_tokens={MAX_TOKENS * 2}")
                        retry_response = retry_llm.invoke(messages)
                        
                        # Extract metadata from retry using helper method
                        retry_response_metadata, retry_usage_metadata, finish_reason = self._extract_usage_metadata(retry_response)
                        
                        # Log retry token usage if available
                        if retry_usage_metadata:
                            input_tokens = retry_usage_metadata.get('input_tokens', 0)
                            output_tokens = retry_usage_metadata.get('output_tokens', 0)
                            total_tokens = retry_usage_metadata.get('total_tokens', 0)
                            if input_tokens > 0 or output_tokens > 0 or total_tokens > 0:
                                logger.info(f"   📊 Retry token usage: input={input_tokens}, output={output_tokens}, total={total_tokens}")
                        
                        # Re-extract content from retry response using helper method
                        generated_text = self._extract_content_from_response(retry_response)
                        
                        if finish_reason == 'MAX_TOKENS':
                            logger.warning(f"   ⚠️ Still hitting MAX_TOKENS after retry with {MAX_TOKENS * 2} tokens")
                        else:
                            logger.info(f"   ✅ Retry successful, finish_reason: {finish_reason}")
                    elif is_complete:
                        logger.info(f"   ✅ Content appears complete despite MAX_TOKENS finish reason")
                    else:
                        logger.warning(f"   ⚠️ Max retries ({max_retries}) reached for MAX_TOKENS, using truncated content")
                
                logger.debug(f"   LLM response received: {len(generated_text) if generated_text else 0} chars")
                if generated_text:
                    logger.debug(f"   Response preview: {generated_text[:200]}...")
                else:
                    logger.warning(f"   ⚠️ Empty response after all extraction attempts!")
                    logger.warning(f"   Response type: {type(response)}")
                    logger.warning(f"   Response repr: {repr(response)[:500]}")
                    if hasattr(response, '__dict__'):
                        logger.warning(f"   Response __dict__ keys: {list(response.__dict__.keys())}")
                        for key in list(response.__dict__.keys())[:10]:
                            try:
                                value = getattr(response, key)
                                logger.warning(f"   {key}: {type(value)} = {str(value)[:100] if value else 'None'}")
                            except:
                                pass
                    
            except Exception as invoke_error:
                logger.error(f"❌ LLM invocation failed: {invoke_error}", exc_info=True)
                generated_text = None
            
            # Ensure we have content
            if not generated_text or len(generated_text.strip()) == 0:
                logger.warning("⚠️ LLM returned empty response, using tone-adjusted content")
                generated_text = state.get("tone_adjusted_content", "")
                if not generated_text:
                    logger.warning("⚠️ Tone-adjusted content also empty, using template")
                    generated_text = state.get("email_template", {}).get("email_template", "")
                    if not generated_text:
                        logger.warning("⚠️ Template also empty, using fallback")
                        generated_text = f"Subject: Professional Correspondence\n\nDear Recipient,\n\n{state['prompt']}\n\nBest regards,\n[Your Name]"
            
            state["generated_email"] = generated_text
            logger.info(f"✅ Email generated: {len(generated_text)} chars")
            logger.debug(f"   Email preview: {generated_text[:200]}...")
            
        except Exception as e:
            logger.error(f"❌ Email generation failed: {e}", exc_info=True)
            logger.debug(f"   Exception type: {type(e)}, Details: {str(e)}")
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
        
        logger.info(f"   Current attempt: {attempt}, Max retries: {EVALUATOR_MAX_RETRIES}")
        logger.info(f"   Previous score: {evaluation.get('score', 0):.1f}, Threshold: {EVALUATOR_THRESHOLD}")
        logger.info(f"   Passed: {evaluation.get('passed', False)}")
        
        if attempt >= EVALUATOR_MAX_RETRIES:
            logger.warning(f"⚠️ Max retries ({EVALUATOR_MAX_RETRIES}) reached, stopping regeneration")
            logger.warning(f"   Final score: {evaluation.get('score', 0):.1f}")
            return state
        
        # Increment attempt
        state["attempt"] = attempt + 1
        logger.info(f"   Incremented attempt to: {state['attempt']}")
        
        # Add feedback to prompt for regeneration
        feedback = evaluation.get("feedback", "")
        critical_errors = evaluation.get("critical_errors", [])
        score = evaluation.get("score", 0)
        
        logger.debug(f"   Feedback length: {len(feedback)} chars")
        logger.debug(f"   Critical errors count: {len(critical_errors)}")
        
        if feedback or critical_errors:
            feedback_text = f"\n\n=== CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT (Score: {score:.1f}/10) ===\n"
            if critical_errors:
                feedback_text += f"CRITICAL ERRORS TO FIX:\n"
                for i, error in enumerate(critical_errors[:5], 1):
                    feedback_text += f"{i}. {error}\n"
                feedback_text += "\n"
            if feedback:
                feedback_text += f"DETAILED FEEDBACK:\n{feedback}\n\n"
            feedback_text += "IMPORTANT INSTRUCTIONS:\n"
            feedback_text += "- You MUST address ALL the issues listed above\n"
            feedback_text += "- Do NOT use placeholders like [Your Time Zone], [Your Name], etc.\n"
            feedback_text += "- Use specific, complete information from the original request\n"
            feedback_text += "- Ensure all dates, times, names, and details are accurate and complete\n"
            feedback_text += "- Generate a complete, ready-to-send email with no placeholders\n"
            
            # Update prompt with feedback (append to original prompt, not replace)
            original_prompt = state.get("prompt", "")
            # Remove any previous feedback to avoid duplication
            if "=== CRITICAL FEEDBACK" in original_prompt:
                original_prompt = original_prompt.split("=== CRITICAL FEEDBACK")[0].strip()
            
            state["prompt"] = original_prompt + feedback_text
            logger.info(f"   ✅ Added feedback: {len(feedback_text)} chars")
            logger.debug(f"   Updated prompt length: {len(state['prompt'])} chars")
            logger.debug(f"   Feedback preview: {feedback_text[:200]}...")
        else:
            logger.warning(f"   ⚠️ No feedback or critical errors to add!")
        
        # Slightly increase temperature for regeneration to encourage variation
        current_temp = state.get("temperature", TEMPERATURE)
        new_temp = min(current_temp + (attempt * 0.1), 1.0)
        state["temperature"] = new_temp
        logger.debug(f"   Temperature adjusted: {current_temp:.2f} → {new_temp:.2f}")
        
        return state
    
    def _should_regenerate(self, state: EmailAgentState) -> str:
        """Determine if email should be regenerated."""
        logger.debug("🔍 Checking if regeneration is needed...")
        
        if not USE_EVALUATOR:
            logger.debug("   Evaluator disabled, ending workflow")
            return "end"
        
        evaluation = state.get("evaluation", {})
        attempt = state.get("attempt", 0)
        passed = evaluation.get("passed", False)
        score = evaluation.get("score", 0)
        
        logger.debug(f"   Evaluation passed: {passed}")
        logger.debug(f"   Score: {score:.1f}, Threshold: {EVALUATOR_THRESHOLD}")
        logger.debug(f"   Current attempt: {attempt}, Max retries: {EVALUATOR_MAX_RETRIES}")
        
        # Regenerate if not passed and haven't exceeded max retries
        if not passed and attempt < EVALUATOR_MAX_RETRIES:
            logger.info(f"🔄 Will regenerate (attempt {attempt + 1}/{EVALUATOR_MAX_RETRIES})")
            logger.info(f"   Reason: Score {score:.1f} < threshold {EVALUATOR_THRESHOLD}")
            if evaluation.get("critical_errors"):
                logger.info(f"   Critical errors: {len(evaluation['critical_errors'])} issues to fix")
            return "regenerate"
        elif attempt >= EVALUATOR_MAX_RETRIES:
            logger.warning(f"⚠️ Max retries reached ({EVALUATOR_MAX_RETRIES}), ending workflow")
            logger.warning(f"   Final score: {score:.1f}")
            return "end"
        elif passed:
            logger.info(f"✅ Email passed evaluation (score: {score:.1f}), ending workflow")
            return "end"
        else:
            logger.warning(f"⚠️ Unexpected state: passed={passed}, attempt={attempt}")
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
