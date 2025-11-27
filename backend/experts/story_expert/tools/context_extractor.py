"""
Context Extractor - Extracts story requirements from user prompts
"""
import os
import logging
import json
from typing import Dict, Any, Optional
import google.generativeai as genai
from ..prompts import CONTEXT_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class ContextExtractor:
    """Extracts story context and requirements from user prompts."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                for model_name in ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-1.5-pro-latest"]:
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
        """Extract story context from prompt."""
        logger.info(f"🔍 Extracting story context from prompt ({len(prompt)} chars)")
        
        if not self.enabled:
            return self._fallback_extract(prompt, enhanced_query)
        
        extraction_prompt = CONTEXT_EXTRACTION_PROMPT.format(prompt=prompt)
        
        try:
            response = self.model.generate_content(extraction_prompt)
            result_text = response.text.strip()
            
            # Remove markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            extracted_data = json.loads(result_text)
            
            logger.info(f"✅ Context extracted: Genre={extracted_data.get('genre')}, Tone={extracted_data.get('tone')}")
            return extracted_data
            
        except Exception as e:
            logger.error(f"❌ Context extraction failed: {e}")
            return self._fallback_extract(prompt, enhanced_query)
    
    def _fallback_extract(self, prompt: str, enhanced_query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback extraction using keyword matching."""
        logger.info("🔄 Using fallback context extraction")
        
        prompt_lower = prompt.lower()
        
        # Genre detection
        genre_keywords = {
            "fantasy": ["fantasy", "magic", "wizard", "dragon", "elf", "kingdom"],
            "sci-fi": ["sci-fi", "science fiction", "space", "alien", "robot", "future", "spaceship"],
            "romance": ["romance", "love", "relationship", "couple", "dating"],
            "mystery": ["mystery", "detective", "crime", "murder", "clue", "investigate"],
            "horror": ["horror", "scary", "haunted", "ghost", "terror", "nightmare"],
            "adventure": ["adventure", "quest", "journey", "treasure", "explore"]
        }
        
        genre = "general"
        for g, keywords in genre_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                genre = g
                break
        
        # Tone detection
        tone = "creative"
        if any(word in prompt_lower for word in ["dark", "grim", "serious"]):
            tone = "dark"
        elif any(word in prompt_lower for word in ["funny", "humorous", "comedy"]):
            tone = "humorous"
        elif any(word in prompt_lower for word in ["light", "cheerful", "uplifting"]):
            tone = "light"
        
        return {
            "story_type": "short_story",
            "genre": genre,
            "tone": tone,
            "themes": [],
            "setting": {"time_period": "unspecified", "location": "unspecified", "atmosphere": tone},
            "characters": {"protagonist": "to be developed"},
            "plot_elements": {"conflict": "to be developed"},
            "style_preferences": {"pov": "third", "tense": "past"},
            "length_target": "medium",
            "special_requirements": []
        }