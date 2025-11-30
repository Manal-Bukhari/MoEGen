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
from utils.json_parser import parse_json_robust

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
                    model_name,  # Try user-specified model first
                    "gemini-2.5-flash",
                    "gemini-2.0-flash-exp",
                    "gemini-1.5-flash",
                    "gemini-pro"
                ]
                
                self.model = None
                self.model_name = None
                for model_to_try in model_names_to_try:
                    if not model_to_try:
                        continue
                    try:
                        self.model = genai.GenerativeModel(model_to_try)
                        self.model_name = model_to_try
                        logger.info(f"✅ Query Enhancer initialized with Gemini: {model_to_try}")
                        self.use_gemini = True
                        break
                    except (ValueError, AttributeError, RuntimeError) as e:
                        logger.debug(f"   Model {model_to_try} failed: {str(e)[:100]}")
                        continue
                
                if not self.model:
                    raise RuntimeError("No Gemini models available - all model attempts failed")
                    
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.warning(f"⚠️ Gemini init failed: {e}. Using fallback.")
                self.model = None
                self.model_name = None
                self.use_gemini = False
        else:
            logger.warning("⚠️ GEMINI_API_KEY not set. Using fallback.")
            self.model = None
            self.model_name = None
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
            # Generate content with Gemini
            logger.debug(f"   Sending prompt to Gemini ({self.model_name})...")
            response = self.model.generate_content(
                enhancement_prompt,
                generation_config={
                    "temperature": 0.3,  # Lower temperature for more consistent JSON output
                    "top_p": 0.95,
                    "top_k": 40
                }
            )
            
            if not response or not hasattr(response, 'text'):
                raise ValueError("Empty or invalid response from Gemini")
                
            result_text = response.text.strip()
            logger.debug(f"   Received response ({len(result_text)} chars)")
            
            # Use robust JSON parser
            logger.debug("🔧 Parsing JSON with robust parser...")
            try:
                enhanced_data = parse_json_robust(result_text)
                logger.debug(f"   Successfully parsed JSON with keys: {list(enhanced_data.keys())}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"❌ JSON parsing failed: {e}")
                logger.debug(f"   Problematic JSON (first 500 chars): {result_text[:500]}")
                # Try to extract any useful information before falling back
                raise
            
            # Validate and enrich enhanced data
            enhanced_data['original_query'] = user_query
            enhanced_data['expert_type'] = expert_type or 'auto'
            
            # Ensure required fields exist with defaults
            if expert_type == "email":
                enhanced_data.setdefault('email_type', 'general')
                enhanced_data.setdefault('tone', 'professional')
                enhanced_data.setdefault('key_points', [])
                enhanced_data.setdefault('recipient_type', 'general')
                enhanced_data.setdefault('special_requirements', [])
                enhanced_data.setdefault('enhanced_instruction', user_query)
            elif expert_type == "story":
                enhanced_data.setdefault('genre', 'general')
                enhanced_data.setdefault('tone', 'creative')
                enhanced_data.setdefault('key_elements', [])
                enhanced_data.setdefault('length_preference', 'medium')
                enhanced_data.setdefault('special_requirements', [])
                enhanced_data.setdefault('enhanced_instruction', user_query)
            elif expert_type == "poem":
                enhanced_data.setdefault('poem_type', 'free_verse')
                enhanced_data.setdefault('tone', 'expressive')
                enhanced_data.setdefault('theme', 'general')
                enhanced_data.setdefault('rhyme_scheme', 'free_verse')
                enhanced_data.setdefault('special_requirements', [])
                enhanced_data.setdefault('enhanced_instruction', user_query)
            
            logger.info(f"✅ Enhanced query for {enhanced_data.get('expert_type', 'unknown')} expert")
            logger.debug("   Enhanced query structure:")
            if expert_type == "email":
                logger.debug(f"     - email_type: {enhanced_data.get('email_type', 'N/A')}")
                logger.debug(f"     - tone: {enhanced_data.get('tone', 'N/A')}")
                logger.debug(f"     - recipient_type: {enhanced_data.get('recipient_type', 'N/A')}")
                logger.debug(f"     - key_points: {enhanced_data.get('key_points', [])}")
            logger.debug(f"     - special_requirements: {enhanced_data.get('special_requirements', [])}")
            logger.debug(f"     - enhanced_instruction: {enhanced_data.get('enhanced_instruction', 'N/A')[:100]}...")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ Enhancement failed: {e}", exc_info=True)
            logger.warning("   Falling back to rule-based enhancement")
            return self._fallback_enhancement(user_query, expert_type)
    
    def _create_email_prompt(self, user_query: str) -> str:
        """Create enhancement prompt for email expert."""
        return f"""You are an expert at analyzing email requests and extracting structured information.

TASK: Analyze the following email request and extract key information to help generate a professional email.

USER REQUEST:
"{user_query}"

INSTRUCTIONS:
1. Identify the email type (sick_leave, vacation, meeting, thank_you, inquiry, complaint, resignation, invitation, follow_up, general, etc.)
2. Determine the appropriate tone (formal, semi-formal, casual, professional, friendly)
3. Extract all key points that need to be included in the email
4. Identify the recipient type (HR, manager, boss, supervisor, client, team, colleague, general)
5. Note any special requirements or constraints
6. Create a detailed, enhanced instruction that expands on the user's request with context and clarity

IMPORTANT: Return ONLY valid JSON. Do not include any markdown formatting, code blocks, or explanatory text. Just the raw JSON object.

Return this exact JSON structure:
{{
    "email_type": "sick_leave",
    "tone": "formal",
    "key_points": ["point1", "point2"],
    "recipient_type": "HR",
    "special_requirements": ["requirement1"],
    "enhanced_instruction": "A detailed, expanded instruction that provides full context for generating the email"
}}

JSON:"""
    
    def _create_story_prompt(self, user_query: str) -> str:
        """Create enhancement prompt for story expert."""
        return f"""You are an expert at analyzing story requests and extracting structured information for creative writing.

TASK: Analyze the following story request and extract key information to help generate a compelling story.

USER REQUEST:
"{user_query}"

INSTRUCTIONS:
1. Identify the genre (fantasy, sci-fi, mystery, thriller, romance, horror, adventure, drama, comedy, general, etc.)
2. Determine the tone (serious, humorous, dark, light, suspenseful, emotional, whimsical, etc.)
3. Extract key elements: characters mentioned, settings implied, themes, plot elements
4. Determine length preference (short: 500-1000 words, medium: 1000-2000 words, long: 2000+ words)
5. Note any special requirements (specific characters, settings, plot twists, style preferences)
6. Create a detailed, enhanced instruction that expands on the user's request with creative context

IMPORTANT: Return ONLY valid JSON. Do not include any markdown formatting, code blocks, or explanatory text. Just the raw JSON object.

Return this exact JSON structure:
{{
    "genre": "fantasy",
    "tone": "adventurous",
    "key_elements": ["element1", "element2"],
    "length_preference": "medium",
    "special_requirements": ["requirement1"],
    "enhanced_instruction": "A detailed, expanded instruction that provides full creative context for generating the story"
}}

JSON:"""
    
    def _create_poem_prompt(self, user_query: str) -> str:
        """Create enhancement prompt for poem expert."""
        return f"""You are an expert at analyzing poetry requests and extracting structured information for poetic composition.

TASK: Analyze the following poem request and extract key information to help generate a beautiful poem.

USER REQUEST:
"{user_query}"

INSTRUCTIONS:
1. Identify the poem type (haiku, sonnet, free_verse, limerick, ballad, ode, acrostic, etc.)
2. Determine the tone (romantic, melancholic, joyful, serious, contemplative, playful, nostalgic, etc.)
3. Identify the theme (love, nature, life, friendship, loss, hope, beauty, etc.)
4. Determine rhyme scheme preference (rhyming, free_verse, specific_pattern)
5. Note any special requirements (specific meter, imagery, metaphors, length, etc.)
6. Create a detailed, enhanced instruction that expands on the user's request with poetic context

IMPORTANT: Return ONLY valid JSON. Do not include any markdown formatting, code blocks, or explanatory text. Just the raw JSON object.

Return this exact JSON structure:
{{
    "poem_type": "free_verse",
    "tone": "contemplative",
    "theme": "nature",
    "rhyme_scheme": "free_verse",
    "special_requirements": ["requirement1"],
    "enhanced_instruction": "A detailed, expanded instruction that provides full poetic context for generating the poem"
}}

JSON:"""
    
    def _create_generic_prompt(self, user_query: str) -> str:
        """Create generic enhancement prompt that works for all expert types."""
        return f"""You are an expert at analyzing text generation requests and extracting structured information.

TASK: Analyze the following text generation request and extract key information to help generate appropriate content.

USER REQUEST:
"{user_query}"

INSTRUCTIONS:
1. Identify the content type (story, poem, email, general)
2. Determine the appropriate tone based on the request
3. Extract all key points that need to be included
4. Note any special requirements or constraints
5. Create a detailed, enhanced instruction that expands on the user's request with full context

IMPORTANT: Return ONLY valid JSON. Do not include any markdown formatting, code blocks, or explanatory text. Just the raw JSON object.

Return this exact JSON structure:
{{
    "content_type": "general",
    "tone": "appropriate",
    "key_points": ["point1", "point2"],
    "special_requirements": ["requirement1"],
    "enhanced_instruction": "A detailed, expanded instruction that provides full context for generating the content"
}}

JSON:"""
    
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

