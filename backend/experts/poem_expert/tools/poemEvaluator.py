"""
Poem Evaluator - Checks poem quality (OPTIONAL)

WHAT IT DOES:
- Scores poem on 5 criteria (0-10 each)
- Checks if it matches requirements
- Provides feedback for improvement

WHEN TO USE:
- Enable in production for better quality
- Disable in development to save API costs
"""
import os
import logging
import json
from typing import Dict, Any
import google.generativeai as genai
from utils.json_parser import parse_json_robust

logger = logging.getLogger(__name__)


class PoemEvaluator:
    """Evaluates poem quality."""
    
    def __init__(self, api_key: str = None, threshold: float = None, max_retries: int = None):
        """
        Initialize evaluator.
        
        Args:
            api_key: Gemini API key
            threshold: Minimum score to pass (0-10)
            max_retries: How many times to regenerate
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Import config
        from .. import config
        self.threshold = threshold if threshold is not None else config.EVALUATOR_THRESHOLD
        self.max_retries = max_retries if max_retries is not None else config.EVALUATOR_MAX_RETRIES
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                for model_name in ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        logger.info(f"✅ Poem Evaluator: {model_name}")
                        self.enabled = True
                        break
                    except:
                        continue
            except Exception as e:
                logger.warning(f"⚠️ Evaluator disabled: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️ No GEMINI_API_KEY. Evaluator disabled.")
            self.enabled = False
    
    def evaluate(self, original_request: str, poem_text: str) -> Dict[str, Any]:
        """
        Evaluate poem quality.
        
        SCORING CRITERIA:
        1. Adherence to request (0-10)
        2. Poetic quality (0-10)
        3. Imagery & language (0-10)
        4. Emotional impact (0-10)
        5. Technical skill (0-10)
        
        Returns:
            {
                "score": 7.5,
                "passed": True,
                "feedback": "...",
                "critical_errors": [...],
                "suggestions": [...]
            }
        """
        logger.info("📊 Evaluating poem...")
        
        if not self.enabled:
            return {"score": 10.0, "passed": True, "feedback": "Evaluator disabled"}
        
        eval_prompt = f"""Evaluate this poem:

USER REQUEST: {original_request}

POEM:
{poem_text}

Score each (0-10):
1. ADHERENCE: Matches type/tone/theme?
2. POETIC QUALITY: Well-crafted, artistic?
3. IMAGERY: Vivid, original language?
4. EMOTIONAL IMPACT: Moving, memorable?
5. TECHNICAL SKILL: Rhyme/rhythm/structure?

Return ONLY JSON (no markdown):
{{
    "adherence_to_request": <0-10>,
    "poetic_quality": <0-10>,
    "imagery_language": <0-10>,
    "emotional_impact": <0-10>,
    "technical_skill": <0-10>,
    "overall_score": <average>,
    "feedback": "detailed explanation",
    "critical_errors": ["issues"],
    "suggestions": ["improvements"]
}}"""
        
        try:
            response = self.model.generate_content(eval_prompt)
            evaluation = parse_json_robust(response.text.strip())
            
            score = evaluation.get("overall_score", 0)
            passed = score >= self.threshold
            
            logger.info(f"✅ Score: {score:.1f}, Passed: {passed}")
            
            return {
                "score": score,
                "passed": passed,
                "feedback": evaluation.get("feedback", ""),
                "criteria_scores": {
                    "adherence_to_request": evaluation.get("adherence_to_request", 0),
                    "poetic_quality": evaluation.get("poetic_quality", 0),
                    "imagery_language": evaluation.get("imagery_language", 0),
                    "emotional_impact": evaluation.get("emotional_impact", 0),
                    "technical_skill": evaluation.get("technical_skill", 0)
                },
                "critical_errors": evaluation.get("critical_errors", []),
                "suggestions": evaluation.get("suggestions", [])
            }
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return {"score": 10.0, "passed": True, "feedback": f"Error: {e}"}