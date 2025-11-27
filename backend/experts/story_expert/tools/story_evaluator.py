"""
Story Evaluator - Evaluates story quality
"""
import os
import logging
import json
from typing import Dict, Any
import google.generativeai as genai
from ..prompts import STORY_EVALUATION_PROMPT

logger = logging.getLogger(__name__)


class StoryEvaluator:
    """Evaluates story quality against requirements."""
    
    def __init__(self, api_key: str = None, threshold: float = None, max_retries: int = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        from .. import config
        self.threshold = threshold if threshold is not None else config.EVALUATOR_THRESHOLD
        self.max_retries = max_retries if max_retries is not None else config.EVALUATOR_MAX_RETRIES
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                for model_name in ["gemini-2.5-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"]:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        logger.info(f"✅ Story Evaluator: {model_name}")
                        self.enabled = True
                        break
                    except:
                        continue
                if not hasattr(self, 'model'):
                    raise Exception("No Gemini models available")
            except Exception as e:
                logger.warning(f"⚠️ Story Evaluator disabled: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️ No GEMINI_API_KEY. Story Evaluator disabled.")
            self.enabled = False
    
    def evaluate(self, original_request: str, story_text: str) -> Dict[str, Any]:
        """Evaluate story quality."""
        logger.info("📊 Evaluating story quality")
        
        if not self.enabled:
            logger.warning("⚠️ Evaluator disabled, returning default pass")
            return {"score": 10.0, "passed": True, "feedback": "Evaluator disabled"}
        
        # Programmatic checks first
        prog_checks = self._programmatic_checks(original_request, story_text)
        
        # LLM evaluation
        eval_prompt = STORY_EVALUATION_PROMPT.format(
            original_request=original_request,
            story_text=story_text[:3000]  # Truncate for evaluation
        )
        
        try:
            response = self.model.generate_content(eval_prompt)
            result_text = response.text.strip()
            
            # Remove markdown
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(result_text)
            
            llm_score = evaluation.get("overall_score", 0)
            final_score = max(0, llm_score - prog_checks["penalty"])
            
            passed = final_score >= self.threshold
            
            # Combine feedback
            combined_feedback = evaluation.get("feedback", "")
            if prog_checks["issues"]:
                combined_feedback = "CRITICAL ISSUES: " + "; ".join(prog_checks["issues"]) + ". " + combined_feedback
            
            logger.info(f"✅ Evaluation complete: Score={final_score:.1f}, Passed={passed}")
            
            return {
                "score": final_score,
                "llm_score": llm_score,
                "penalty": prog_checks["penalty"],
                "passed": passed,
                "feedback": combined_feedback,
                "criteria_scores": {
                    "adherence_to_request": evaluation.get("adherence_to_request", 0),
                    "story_structure": evaluation.get("story_structure", 0),
                    "character_development": evaluation.get("character_development", 0),
                    "writing_quality": evaluation.get("writing_quality", 0),
                    "emotional_impact": evaluation.get("emotional_impact", 0)
                },
                "critical_errors": evaluation.get("critical_errors", []) + prog_checks["issues"],
                "suggestions": evaluation.get("suggestions", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            
            if not prog_checks["passed"]:
                fallback_score = max(0, 10.0 - prog_checks["penalty"])
                return {
                    "score": fallback_score,
                    "passed": fallback_score >= self.threshold,
                    "feedback": "; ".join(prog_checks["issues"]),
                    "criteria_scores": {},
                    "critical_errors": prog_checks["issues"]
                }
            
            return {"score": 10.0, "passed": True, "feedback": f"Error: {e}"}
    
    def _programmatic_checks(self, request: str, story: str) -> Dict[str, Any]:
        """Run programmatic checks."""
        issues = []
        
        # Check minimum length
        word_count = len(story.split())
        if word_count < 100:
            issues.append(f"Story too short ({word_count} words, minimum 100)")
        
        # Check for incomplete story (common truncation signs)
        if story.rstrip().endswith("...") or story.rstrip().endswith("["):
            issues.append("Story appears incomplete or truncated")
        
        # Check for story structure
        if not self._has_story_structure(story):
            issues.append("Story lacks clear beginning, middle, or end")
        
        penalty = min(len(issues) * 2.0, 6.0)
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "penalty": penalty
        }
    
    def _has_story_structure(self, story: str) -> bool:
        """Check if story has basic structure."""
        paragraphs = [p.strip() for p in story.split("\n\n") if p.strip()]
        return len(paragraphs) >= 3  # At least beginning, middle, end