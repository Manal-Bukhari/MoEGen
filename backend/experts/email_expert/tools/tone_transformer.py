"""
Tone Transformer - Transforms email tone to match requirements
"""
import os
import logging
import json
import re
from typing import Dict, Any
import google.generativeai as genai
from utils.json_parser import parse_json_robust

logger = logging.getLogger(__name__)


class ToneTransformer:
    """Transforms email tone to match formality and style requirements."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Try multiple models for compatibility
                for model_name in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        logger.info(f"✅ Tone Transformer: {model_name}")
                        self.enabled = True
                        break
                    except:
                        continue
                if not hasattr(self, 'model'):
                    raise Exception("No Gemini models available")
            except Exception as e:
                logger.warning(f"⚠️ Tone Transformer disabled: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️ No GEMINI_API_KEY. Tone Transformer disabled.")
            self.enabled = False
    
    def transform(self, email_content: str, extracted_context: Dict[str, Any], enhanced_query: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transform email tone to match requirements.
        
        Args:
            email_content: Email content (template or generated email)
            extracted_context: Context from ContextExtractor
            enhanced_query: Optional enhanced query from router
            
        Returns:
            Dictionary with:
            - tone_adjusted_email: Email with adjusted tone
            - tone_analysis: Analysis of tone adjustments made
        """
        logger.info("🎨 ToneTransformer.transform() called")
        logger.debug(f"   Email content length: {len(email_content)} chars")
        logger.debug(f"   Formality level: {extracted_context.get('formality_level', 'N/A')}")
        
        if not self.enabled:
            logger.warning("⚠️ ToneTransformer disabled, using fallback transformation")
            return self._fallback_transform(email_content, extracted_context, enhanced_query)
        
        # Build tone transformation prompt
        logger.debug("📝 Building tone transformation prompt...")
        tone_prompt = self._create_tone_prompt(email_content, extracted_context, enhanced_query)
        logger.debug(f"   Tone prompt length: {len(tone_prompt)} chars")
        
        try:
            logger.info("🤖 Calling Gemini API for tone transformation...")
            response = self.model.generate_content(tone_prompt)
            result_text = response.text.strip()
            logger.debug(f"   Raw response length: {len(result_text)} chars")
            
            # Use robust JSON parser
            logger.debug("🔧 Parsing JSON with robust parser...")
            try:
                tone_data = parse_json_robust(result_text)
                logger.debug(f"   Successfully parsed JSON with keys: {list(tone_data.keys())}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"❌ JSON parse error: {e}")
                logger.debug(f"   Problematic JSON (first 500 chars): {result_text[:500]}")
                logger.warning("⚠️ Falling back to fallback transformation")
                return self._fallback_transform(email_content, extracted_context, enhanced_query)
            
            # Ensure all required fields exist
            logger.debug("🔍 Validating tone data structure...")
            result = {
                "tone_adjusted_email": tone_data.get("tone_adjusted_email", email_content),
                "tone_analysis": tone_data.get("tone_analysis", "No tone adjustments made")
            }
            
            logger.info(f"✅ Tone transformed successfully:")
            logger.info(f"   Adjusted email length: {len(result['tone_adjusted_email'])} chars")
            logger.debug(f"   Tone analysis: {result['tone_analysis'][:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"❌ Tone transformation failed: {e}", exc_info=True)
            logger.warning("⚠️ Falling back to fallback transformation")
            return self._fallback_transform(email_content, extracted_context, enhanced_query)
    
    def _create_tone_prompt(self, email_content: str, extracted_context: Dict[str, Any], enhanced_query: Dict[str, Any] = None) -> str:
        """Create prompt for tone transformation."""
        formality_level = extracted_context.get("formality_level", "professional")
        email_type = extracted_context.get("email_type", "general")
        recipient = extracted_context.get("key_entities", {}).get("recipient", "general")
        
        target_tone = "professional"
        if enhanced_query:
            target_tone = enhanced_query.get("tone", "professional")
        
        return f"""Adjust the tone of this email to match the requirements:

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
    
    def _fallback_transform(self, email_content: str, extracted_context: Dict[str, Any], enhanced_query: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback tone transformation when LLM is not available."""
        logger.info("🔄 Using fallback tone transformation")
        
        # Simple tone adjustments without LLM
        formality_level = extracted_context.get("formality_level", "professional")
        target_tone = "professional"
        if enhanced_query:
            target_tone = enhanced_query.get("tone", "professional")
        logger.debug(f"   Target tone: {target_tone}, Formality: {formality_level}")
        
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
        logger.debug(f"   Applied basic tone adjustments: {target_tone}")
        
        logger.info(f"✅ Fallback tone transformation complete: {target_tone}")
        return {
            "tone_adjusted_email": adjusted_email,
            "tone_analysis": analysis
        }


