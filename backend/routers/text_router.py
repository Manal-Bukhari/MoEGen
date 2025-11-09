import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TextRouter:
    """
    Router that analyzes input prompts and routes them to the appropriate expert.
    Uses keyword matching and scoring to determine the best expert.
    """
    
    def __init__(self, story_expert, poem_expert, email_expert):
        self.experts = {
            "story": story_expert,
            "poem": poem_expert,
            "email": email_expert
        }
        
        # Define keywords for each expert with weights
        self.expert_keywords = {
            "story": {
                "story": 10,
                "tale": 9,
                "narrative": 8,
                "fiction": 8,
                "adventure": 7,
                "character": 6,
                "plot": 6,
                "once upon": 10,
                "beginning": 5,
                "ending": 5,
                "chapter": 7,
                "novel": 8,
                "short story": 10,
                "fantasy": 7,
                "sci-fi": 7,
                "mystery": 7,
                "thriller": 7
            },
            "poem": {
                "poem": 10,
                "poetry": 10,
                "verse": 9,
                "rhyme": 8,
                "haiku": 10,
                "sonnet": 10,
                "stanza": 8,
                "lyric": 7,
                "ballad": 8,
                "ode": 9,
                "limerick": 9,
                "free verse": 9,
                "metaphor": 6,
                "romantic": 5,
                "poetic": 9
            },
            "email": {
                "email": 10,
                "letter": 8,
                "message": 7,
                "professional": 8,
                "formal": 8,
                "business": 7,
                "write to": 6,
                "send to": 6,
                "dear": 7,
                "sincerely": 8,
                "regards": 7,
                "request": 6,
                "inquiry": 7,
                "application": 7,
                "proposal": 7,
                "meeting": 6,
                "follow up": 7
            }
        }
    
    def calculate_expert_scores(self, prompt: str) -> Dict[str, float]:
        """
        Calculate confidence scores for each expert based on keyword matching.
        Returns a dictionary of expert names to confidence scores (0-1).
        """
        prompt_lower = prompt.lower()
        scores = {expert: 0.0 for expert in self.experts.keys()}
        
        # Calculate weighted keyword scores
        for expert, keywords in self.expert_keywords.items():
            total_weight = 0
            matched_weight = 0
            
            for keyword, weight in keywords.items():
                total_weight += weight
                if keyword in prompt_lower:
                    matched_weight += weight
                    logger.debug(f"Matched '{keyword}' for {expert} expert (weight: {weight})")
            
            # Normalize score
            if total_weight > 0:
                scores[expert] = matched_weight / total_weight
        
        # Add small random noise to break ties
        for expert in scores:
            scores[expert] += np.random.uniform(0, 0.01)
        
        # If no keywords matched, use uniform distribution
        if all(score <= 0.01 for score in scores.values()):
            logger.info("No clear expert match, using default (story)")
            scores["story"] = 0.4
            scores["poem"] = 0.3
            scores["email"] = 0.3
        
        return scores
    
    def select_expert(self, prompt: str, force_expert: Optional[str] = None) -> tuple:
        """
        Select the best expert for the given prompt.
        Returns: (expert_name, confidence_score)
        """
        if force_expert:
            if force_expert in self.experts:
                logger.info(f"Forcing expert: {force_expert}")
                return force_expert, 1.0
            else:
                raise ValueError(f"Unknown expert: {force_expert}")
        
        scores = self.calculate_expert_scores(prompt)
        best_expert = max(scores, key=scores.get)
        confidence = scores[best_expert]
        
        logger.info(f"Expert scores: {scores}")
        logger.info(f"Selected expert: {best_expert} (confidence: {confidence:.2f})")
        
        return best_expert, confidence
    
    def route_and_generate(
        self, 
        prompt: str, 
        max_length: int = 150,
        temperature: float = 0.7,
        force_expert: Optional[str] = None
    ) -> Dict:
        """
        Route the prompt to the appropriate expert and generate text.
        Returns a dictionary with generated text, expert used, and confidence.
        """
        # Select the expert
        expert_name, confidence = self.select_expert(prompt, force_expert)
        expert = self.experts[expert_name]
        
        # Generate text using the selected expert
        generated_text = expert.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        return {
            "generated_text": generated_text,
            "expert": expert_name,
            "confidence": float(confidence),
            "all_scores": self.calculate_expert_scores(prompt)
        }
