import os
import logging
from typing import Dict, Any
import google.generativeai as genai

logger = logging.getLogger(__name__)

class QueryEnhancer:
    """Enhances queries using Gemini."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        model_name = model_name or os.getenv("QUERY_ENHANCER_MODEL", "gemini-2.5-flash")
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                
                model_names_to_try = [
                    "gemini-2.5-flash",
                    "gemini-2.0-flash",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
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
    
    def enhance(self, user_query: str) -> Dict[str, Any]:
        logger.info(f"🔍 Enhancing: {user_query[:100]}...")
        
        if not self.use_gemini:
            return self._fallback_enhancement(user_query)
        
        enhancement_prompt = f"""Analyze this email request:

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

        try:
            response = self.model.generate_content(enhancement_prompt)
            result_text = response.text.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            # Clean JSON aggressively
            import re
            import json
            
            # Remove newlines and normalize whitespace
            result_text = ' '.join(result_text.split())
            
            # Remove trailing commas
            result_text = re.sub(r',(\s*[}\]])', r'\1', result_text)
            
            # Fix common issues: single quotes to double quotes (if not escaped)
            result_text = result_text.replace("'", '"')
            
            # Remove any non-JSON text before first { and after last }
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
            
            logger.info(f"✅ Enhanced: {enhanced_data.get('email_type', 'unknown')}")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ Enhancement failed: {e}")
            return self._fallback_enhancement(user_query)
    
    def _fallback_enhancement(self, user_query: str) -> Dict[str, Any]:
        logger.warning("Using fallback")
        
        query_lower = user_query.lower()
        
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
            "original_query": user_query
        }