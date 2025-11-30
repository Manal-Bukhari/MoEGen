"""
Tone Transformer - Transforms email tone to match requirements
"""
import logging
from typing import Dict, Any
from .base_tool import init_gemini_model, parse_json_response

logger = logging.getLogger(__name__)


class ToneTransformer:
    """Transforms email tone to match formality and style requirements."""
    
    def __init__(self, api_key: str):
        """
        Initialize ToneTransformer.
        
        Args:
            api_key: Gemini API key (required)
        """
        self.api_key = api_key
        
        # Initialize model using shared utility
        self.model = init_gemini_model(self.api_key)
        self.enabled = self.model is not None
        
        if not self.enabled:
            logger.warning("Tone Transformer disabled: model initialization failed")
    
    def transform(self, email_content: str, extracted_context: Dict[str, Any], enhanced_query: Dict[str, Any] = None, original_prompt: str = None) -> Dict[str, Any]:
        """
        Transform email tone to match requirements.
        
        Args:
            email_content: Email content (template or generated email)
            extracted_context: Context from ContextExtractor
            enhanced_query: Optional enhanced query from router
            original_prompt: Original user query/prompt (for better context understanding)
            
        Returns:
            Dictionary with:
            - tone_adjusted_email: Email with adjusted tone
            - tone_analysis: Analysis of tone adjustments made
        """
        logger.debug(f"ToneTransformer.transform() called: email_length={len(email_content)} chars, formality={extracted_context.get('formality_level', 'N/A')}")
        
        if not self.enabled:
            logger.warning("ToneTransformer disabled, using fallback transformation")
            return self._fallback_transform(email_content, extracted_context, enhanced_query, original_prompt)
        
        # Build tone transformation prompt
        tone_prompt = self._create_tone_prompt(email_content, extracted_context, enhanced_query, original_prompt)
        
        try:
            logger.info("Calling Gemini API for tone transformation...")
            response = self.model.generate_content(tone_prompt)
            result_text = response.text.strip()
            
            # Use shared JSON parser utility
            tone_data = parse_json_response(result_text, "tone transformation")
            if not tone_data:
                logger.warning("Falling back to fallback transformation")
                return self._fallback_transform(email_content, extracted_context, enhanced_query, original_prompt)
            
            # Ensure all required fields exist
            result = {
                "tone_adjusted_email": tone_data.get("tone_adjusted_email", email_content),
                "tone_analysis": tone_data.get("tone_analysis", "No tone adjustments made")
            }
            
            logger.info(f"Tone transformed: adjusted_email_length={len(result['tone_adjusted_email'])} chars")
            return result
            
        except Exception as e:
            logger.error(f"Tone transformation failed: {e}", exc_info=True)
            logger.warning("Falling back to fallback transformation")
            return self._fallback_transform(email_content, extracted_context, enhanced_query, original_prompt)
    
    def _create_tone_prompt(self, email_content: str, extracted_context: Dict[str, Any], enhanced_query: Dict[str, Any] = None, original_prompt: str = None) -> str:
        """Create prompt for tone transformation."""
        formality_level = extracted_context.get("formality_level", "professional")
        email_type = extracted_context.get("email_type", "general")
        recipient = extracted_context.get("key_entities", {}).get("recipient", "general")
        
        target_tone = "professional"
        if enhanced_query:
            target_tone = enhanced_query.get("tone", "professional")
        
        original_request_section = ""
        if original_prompt:
            original_request_section = f"""
ORIGINAL USER REQUEST:
{original_prompt}
"""
        
        return f"""Adjust the tone of this email to match the requirements:
{original_request_section}
ORIGINAL EMAIL:
{email_content}

CONTEXT:
- Email Type: {email_type}
- Recipient: {recipient}
- Current Formality: {formality_level}
- Target Tone: {target_tone}

Transform the email tone to be {target_tone} and appropriate for a {recipient} recipient. 
Maintain all specific details (dates, names, facts) but adjust:
- Formality level (formal/semi-formal/casual)
- Politeness markers
- Sentence structure
- Word choice
- Overall style

Return ONLY valid JSON:
{{
    "tone_adjusted_email": "Email with adjusted tone, maintaining all specific details",
    "tone_analysis": "Brief explanation of tone adjustments made (e.g., 'Made more formal by using professional language and formal greetings')"
}}

Ensure the tone is consistent throughout the email."""
    
    def _fallback_transform(self, email_content: str, extracted_context: Dict[str, Any], enhanced_query: Dict[str, Any] = None, original_prompt: str = None) -> Dict[str, Any]:
        """Fallback tone transformation when LLM is not available."""
        logger.info("Using fallback tone transformation")
        
        # Simple tone adjustments without LLM
        formality_level = extracted_context.get("formality_level", "professional")
        target_tone = "professional"
        if enhanced_query:
            target_tone = enhanced_query.get("tone", "professional")
        
        adjusted_email = email_content
        
        # Basic tone adjustments
        if target_tone == "formal" or formality_level == "formal":
            # Make more formal
            adjusted_email = adjusted_email.replace("Hi ", "Dear ")
            adjusted_email = adjusted_email.replace("Hello ", "Dear ")
            adjusted_email = adjusted_email.replace("Thanks", "Thank you")
            adjusted_email = adjusted_email.replace("I'm", "I am")
            adjusted_email = adjusted_email.replace("I'll", "I will")
            adjusted_email = adjusted_email.replace("can't", "cannot")
        elif target_tone == "casual":
            # Make more casual
            adjusted_email = adjusted_email.replace("Dear ", "Hi ")
            adjusted_email = adjusted_email.replace("Thank you", "Thanks")
            adjusted_email = adjusted_email.replace("I am", "I'm")
            adjusted_email = adjusted_email.replace("I will", "I'll")
            adjusted_email = adjusted_email.replace("cannot", "can't")
        
        analysis = f"Adjusted tone to {target_tone} level"
        
        logger.info(f"Fallback tone transformation complete: {target_tone}")
        return {
            "tone_adjusted_email": adjusted_email,
            "tone_analysis": analysis
        }


