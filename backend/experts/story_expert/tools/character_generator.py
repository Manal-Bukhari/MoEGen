"""
Character Generator - Creates character profiles
"""
import os
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)


class CharacterGenerator:
    """Generates character profiles for stories."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                for model_name in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        logger.info(f"✅ Character Generator: {model_name}")
                        self.enabled = True
                        break
                    except:
                        continue
            except Exception as e:
                logger.warning(f"⚠️ Character Generator disabled: {e}")
                self.enabled = False
        else:
            self.enabled = False
    
    def generate(self, archetype: str, genre: str) -> str:
        """Generate character profile."""
        if not self.enabled:
            return f"A {archetype} character in a {genre} story"
        
        prompt = f"""Create a compelling character for a {genre} story.
        
Archetype: {archetype}

Provide:
1. Name and appearance
2. Core personality traits
3. Primary motivation
4. Key flaw or weakness
5. Background/backstory (brief)

Write 2-3 paragraphs."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return f"A {archetype} character in a {genre} story"