"""
Context Extractor - Extracts poem requirements from user prompts

WHAT IT DOES:
- Analyzes user's request
- Identifies poem type (haiku, sonnet, free verse)
- Extracts tone (sad, joyful, romantic)
- Identifies theme (love, nature, life)
- Determines rhyme scheme

HOW IT WORKS:
1. Sends prompt to Gemini
2. Asks Gemini to extract requirements as JSON
3. Parses the JSON response
4. Falls back to keyword matching if API fails
"""
import os
import logging
import json
from typing import Dict, Any, Optional
import google.generativeai as genai
from utils.json_parser import parse_json_robust

logger = logging.getLogger(__name__)


class ContextExtractor:
    """Extracts poem context from user prompts."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the context extractor.
        
        Args:
            api_key: Gemini API key (optional, reads from env)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            try:
                # Configure Gemini
                genai.configure(api_key=self.api_key)
                
                # Try multiple models for compatibility
                for model_name in ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        logger.info(f"✅ Context Extractor: {model_name}")
                        self.enabled = True
                        break
                    except:
                        continue
                
                if not hasattr(self, 'model'):
                    raise Exception("No Gemini models available")
            except Exception as e:
                logger.warning(f"⚠️ Context Extractor disabled: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️ No GEMINI_API_KEY. Context Extractor disabled.")
            self.enabled = False
    
    def extract(self, prompt: str, enhanced_query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract poem context from user prompt.
        
        EXAMPLE:
        Input: "Write a sad poem about lost love"
        Output: {
            "poem_type": "free_verse",
            "tone": "melancholic",
            "theme": "lost love",
            "rhyme_scheme": "free_verse"
        }
        
        Args:
            prompt: User's request
            enhanced_query: Optional enhanced query from router
            
        Returns:
            Dictionary with poem requirements
        """
        logger.info(f"🔍 Extracting context from: {prompt[:100]}...")
        
        # If API is disabled, use fallback
        if not self.enabled:
            return self._fallback_extract(prompt, enhanced_query)
        
        # Build extraction prompt
        extraction_prompt = self._create_extraction_prompt(prompt, enhanced_query)
        
        try:
            # Call Gemini API
            response = self.model.generate_content(extraction_prompt)
            result_text = response.text.strip()
            
            # Parse JSON (handles markdown code blocks)
            extracted_data = parse_json_robust(result_text)
            
            logger.info(f"✅ Extracted: {extracted_data.get('poem_type')}, {extracted_data.get('tone')}")
            return extracted_data
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return self._fallback_extract(prompt, enhanced_query)
    
    def _create_extraction_prompt(self, prompt: str, enhanced_query: Optional[Dict[str, Any]] = None) -> str:
        """Create prompt for Gemini to extract poem requirements."""
        
        enhanced_info = ""
        if enhanced_query:
            enhanced_info = f"""
ENHANCED QUERY INFO:
- Type: {enhanced_query.get('poem_type', 'unknown')}
- Tone: {enhanced_query.get('tone', 'unknown')}
- Theme: {enhanced_query.get('theme', 'unknown')}
"""
        
        return f"""Analyze this poem request and extract requirements:

USER REQUEST: {prompt}
{enhanced_info}

Extract and return ONLY valid JSON (no markdown, no backticks):
{{
    "poem_type": "haiku/sonnet/free_verse/limerick/ballad/acrostic/couplet/etc",
    "tone": "romantic/melancholic/joyful/serious/whimsical/dark/hopeful/etc",
    "theme": "love/nature/life/death/time/beauty/loss/hope/etc",
    "rhyme_scheme": "rhyming/free_verse/ABAB/AABB/ABCB/etc",
    "length_preference": "short/medium/long",
    "special_requirements": ["meter requirements", "specific style", "etc"],
    "imagery_focus": ["visual", "auditory", "tactile", "olfactory"]
}}

Be thorough and extract all poetic elements."""
    
    def _fallback_extract(self, prompt: str, enhanced_query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fallback: Use keyword matching if API fails.
        
        WHY FALLBACK?
        - API might be down
        - API key might be invalid
        - Better to return something than crash!
        """
        logger.info("🔄 Using fallback extraction")
        prompt_lower = prompt.lower()
        
        # Detect poem type
        poem_type = "free_verse"  # default
        if "haiku" in prompt_lower:
            poem_type = "haiku"
        elif "sonnet" in prompt_lower:
            poem_type = "sonnet"
        elif "limerick" in prompt_lower:
            poem_type = "limerick"
        elif "ballad" in prompt_lower:
            poem_type = "ballad"
        
        # Detect tone
        tone = "expressive"
        if any(word in prompt_lower for word in ["sad", "melancholic", "sorrow", "grief"]):
            tone = "melancholic"
        elif any(word in prompt_lower for word in ["happy", "joyful", "cheerful", "glad"]):
            tone = "joyful"
        elif any(word in prompt_lower for word in ["romantic", "love", "passion"]):
            tone = "romantic"
        elif any(word in prompt_lower for word in ["dark", "grim", "horror", "eerie"]):
            tone = "dark"
        
        # Detect theme
        theme = "general"
        if "love" in prompt_lower:
            theme = "love"
        elif "nature" in prompt_lower:
            theme = "nature"
        elif any(word in prompt_lower for word in ["life", "living"]):
            theme = "life"
        elif "death" in prompt_lower:
            theme = "death"
        elif "time" in prompt_lower:
            theme = "time"
        
        # Rhyme scheme
        rhyme_scheme = "free_verse"
        if any(word in prompt_lower for word in ["rhyme", "rhyming"]):
            rhyme_scheme = "rhyming"
        
        # Override with enhanced_query if available
        if enhanced_query:
            poem_type = enhanced_query.get('poem_type', poem_type)
            tone = enhanced_query.get('tone', tone)
            theme = enhanced_query.get('theme', theme)
        
        return {
            "poem_type": poem_type,
            "tone": tone,
            "theme": theme,
            "rhyme_scheme": rhyme_scheme,
            "length_preference": "medium",
            "special_requirements": [],
            "imagery_focus": []
        }