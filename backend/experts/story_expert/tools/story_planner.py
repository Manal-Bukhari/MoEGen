"""
Story Planner - Creates detailed story outlines
"""
import os
import logging
import json
from typing import Dict, Any
import google.generativeai as genai
from ..prompts import STORY_PLANNING_PROMPT

logger = logging.getLogger(__name__)


class StoryPlanner:
    """Generates detailed story plans and outlines."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                for model_name in ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        logger.info(f"✅ Story Planner: {model_name}")
                        self.enabled = True
                        break
                    except:
                        continue
                if not hasattr(self, 'model'):
                    raise Exception("No Gemini models available")
            except Exception as e:
                logger.warning(f"⚠️ Story Planner disabled: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️ No GEMINI_API_KEY. Story Planner disabled.")
            self.enabled = False
    
    def plan(self, extracted_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate story plan from context."""
        logger.info("📋 Planning story structure")
        
        if not self.enabled:
            return self._fallback_plan(extracted_context)
        
        planning_prompt = STORY_PLANNING_PROMPT.format(
            extracted_context=json.dumps(extracted_context, indent=2),
            genre=extracted_context.get("genre", "general"),
            tone=extracted_context.get("tone", "creative")
        )
        
        try:
            response = self.model.generate_content(planning_prompt)
            result_text = response.text.strip()
            
            # Remove markdown
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            plan_data = json.loads(result_text)
            
            logger.info("✅ Story plan created successfully")
            return plan_data
            
        except Exception as e:
            logger.error(f"❌ Story planning failed: {e}")
            return self._fallback_plan(extracted_context)
    
    def _fallback_plan(self, extracted_context: Dict[str, Any]) -> Dict[str, Any]:
        """Simple fallback plan."""
        logger.info("🔄 Using fallback story planning")
        
        return {
            "story_structure": {
                "opening_hook": "Introduce protagonist in their ordinary world",
                "rising_action": ["Inciting incident occurs", "Protagonist faces challenges", "Stakes are raised"],
                "climax": "Protagonist confronts main conflict",
                "falling_action": "Consequences unfold",
                "conclusion": "New equilibrium established"
            },
            "character_arcs": {
                "protagonist_journey": "Growth through adversity"
            },
            "key_scenes": [],
            "thematic_elements": extracted_context.get("themes", []),
            "estimated_word_count": 1500
        }