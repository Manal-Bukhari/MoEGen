"""
Shared Query Enhancer Service
Enhances user queries using Gemini for all expert types (story, poem, email).
"""
import os
import logging
import re
import json
from typing import Dict, Any
import google.generativeai as genai

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """Enhances queries using Gemini for all expert types."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        model_name = model_name or os.getenv("QUERY_ENHANCER_MODEL", "gemini-2.5-flash")
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                
                model_names_to_try = [
                    "gemini-2.5-flash",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "gemini-2.0-flash",
                    model_name
                ]
                
                self.model = None
                for model_to_try in model_names_to_try:
                    try:
                        self.model = genai.GenerativeModel(model_to_try)
                        logger.info(f"✅ Query Enhancer: {model_to_try}")
                        self.use_gemini = True
                        break
                    except:
                        continue
                
                if not self.model:
                    raise Exception("No Gemini models available")
                    
            except Exception as e:
                logger.warning(f"⚠️ Gemini init failed: {e}. Using fallback.")
                self.model = None
                self.use_gemini = False
        else:
            logger.warning("⚠️ GEMINI_API_KEY not set. Using fallback.")
            self.model = None
            self.use_gemini = False
    
    def enhance(self, user_query: str, expert_type: str = None) -> Dict[str, Any]:
        """
        Enhance user query for the specified expert type.
        
        Args:
            user_query: The original user query
            expert_type: Type of expert (story, poem, email) - optional, will be inferred if not provided
            
        Returns:
            Dictionary with enhanced query information
        """
        logger.info(f"🔍 Enhancing query for {expert_type or 'auto'} expert: {user_query[:100]}...")
        
        if not self.use_gemini:
            return self._fallback_enhancement(user_query, expert_type)
        
        # Create expert-specific enhancement prompt
        if expert_type == "email":
            enhancement_prompt = self._create_email_prompt(user_query)
        elif expert_type == "story":
            enhancement_prompt = self._create_story_prompt(user_query)
        elif expert_type == "poem":
            enhancement_prompt = self._create_poem_prompt(user_query)
        else:
            # Generic prompt that works for all types
            enhancement_prompt = self._create_generic_prompt(user_query)
        
        try:
            response = self.model.generate_content(enhancement_prompt)
            result_text = response.text.strip()
            
            # Extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            # Clean JSON
            result_text = ' '.join(result_text.split())
            result_text = re.sub(r',(\s*[}\]])', r'\1', result_text)
            result_text = result_text.replace("'", '"')
            
            if '{' in result_text and '}' in result_text:
                start = result_text.find('{')
                end = result_text.rfind('}') + 1
                result_text = result_text[start:end]
            
            try:
                enhanced_data = json.loads(result_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON error: {e}")
                logger.debug(f"Problematic JSON (first 300 chars): {result_text[:300]}")
                raise
            
            enhanced_data['original_query'] = user_query
            enhanced_data['expert_type'] = expert_type or 'auto'
            
            logger.info(f"✅ Enhanced query for {enhanced_data.get('expert_type', 'unknown')} expert")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ Enhancement failed: {e}")
            return self._fallback_enhancement(user_query, expert_type)
    
    def _create_email_prompt(self, user_query: str) -> str:
        """Create enhancement prompt for email expert."""
        return f"""Analyze this email request:

REQUEST: {user_query}

Return ONLY valid JSON:
{{
    "email_type": "type (sick_leave, vacation, meeting, thank_you, etc.)",
    "tone": "formal/semi-formal/casual",
    "key_points": ["main points"],
    "recipient_type": "HR/manager/client/team/etc",
    "special_requirements": ["requirements"],
    "enhanced_instruction": "detailed instruction"
}}"""
    
    def _create_story_prompt(self, user_query: str) -> str:
        """Create enhancement prompt for story expert."""
        return f"""Analyze this story request:

REQUEST: {user_query}

Return ONLY valid JSON:
{{
    "genre": "fantasy/sci-fi/mystery/thriller/etc",
    "tone": "serious/humorous/dark/light/etc",
    "key_elements": ["characters", "settings", "themes"],
    "length_preference": "short/medium/long",
    "special_requirements": ["requirements"],
    "enhanced_instruction": "detailed instruction"
}}"""
    
    def _create_poem_prompt(self, user_query: str) -> str:
        """Create enhancement prompt for poem expert."""
        return f"""Analyze this poem request:

REQUEST: {user_query}

Return ONLY valid JSON:
{{
    "poem_type": "haiku/sonnet/free_verse/limerick/etc",
    "tone": "romantic/melancholic/joyful/serious/etc",
    "theme": "love/nature/life/etc",
    "rhyme_scheme": "rhyming/free_verse",
    "special_requirements": ["requirements"],
    "enhanced_instruction": "detailed instruction"
}}"""
    
    def _create_generic_prompt(self, user_query: str) -> str:
        """Create generic enhancement prompt that works for all expert types."""
        return f"""Analyze this text generation request:

REQUEST: {user_query}

Return ONLY valid JSON:
{{
    "content_type": "story/poem/email/general",
    "tone": "appropriate tone",
    "key_points": ["main points"],
    "special_requirements": ["requirements"],
    "enhanced_instruction": "detailed instruction"
}}"""
    
    def _fallback_enhancement(self, user_query: str, expert_type: str = None) -> Dict[str, Any]:
        """Fallback enhancement when Gemini is not available."""
        logger.warning("Using fallback enhancement")
        
        query_lower = user_query.lower()
        
        if expert_type == "email":
            email_type = "general"
            if any(word in query_lower for word in ["sick", "ill", "medical"]):
                email_type = "sick_leave"
            elif any(word in query_lower for word in ["vacation", "holiday", "time off"]):
                email_type = "vacation"
            elif any(word in query_lower for word in ["meeting", "schedule"]):
                email_type = "meeting"
            elif any(word in query_lower for word in ["thank", "appreciate"]):
                email_type = "thank_you"
            
            return {
                "email_type": email_type,
                "tone": "formal",
                "key_points": [user_query],
                "recipient_type": "general",
                "special_requirements": [],
                "enhanced_instruction": user_query,
                "original_query": user_query,
                "expert_type": expert_type or "auto"
            }
        elif expert_type == "story":
            return {
                "genre": "general",
                "tone": "creative",
                "key_elements": [user_query],
                "length_preference": "medium",
                "special_requirements": [],
                "enhanced_instruction": user_query,
                "original_query": user_query,
                "expert_type": expert_type or "auto"
            }
        elif expert_type == "poem":
            return {
                "poem_type": "free_verse",
                "tone": "expressive",
                "theme": "general",
                "rhyme_scheme": "free_verse",
                "special_requirements": [],
                "enhanced_instruction": user_query,
                "original_query": user_query,
                "expert_type": expert_type or "auto"
            }
        else:
            return {
                "content_type": "general",
                "tone": "neutral",
                "key_points": [user_query],
                "special_requirements": [],
                "enhanced_instruction": user_query,
                "original_query": user_query,
                "expert_type": expert_type or "auto"
            }

