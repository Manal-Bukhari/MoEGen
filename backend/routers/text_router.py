# routers/text_router.py
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TextRouter:
    """
    Router that analyzes input prompts and routes them to the appropriate expert.
    Uses keyword matching to determine the best expert.
    
    ✅ FIXED: Email expert now uses pipeline defaults (600, 0.5) instead of forcing (300, 0.7)
    """
    
    def __init__(self, story_expert, poem_expert, email_expert):
        self.experts = {
            "story": story_expert,
            "poem": poem_expert,
            "email": email_expert
        }
        
        # ✅ Simple keyword lists
        self.expert_keywords = {
            "story": [
                # Core story words
                "story", "tale", "narrative", "fiction", "novel",
                # Story elements
                "adventure", "character", "plot", "once upon",
                "chapter", "beginning", "ending", "short story",
                # Genres
                "fantasy", "sci-fi", "mystery", "thriller",
                "hero", "villain", "protagonist", "antagonist",
                "magic", "quest", "journey"
            ],
            
            "poem": [
                # Core poetry words
                "poem", "poetry", "verse", "rhyme", "haiku",
                "sonnet", "stanza", "lyric", "ballad", "ode",
                # Poetry styles
                "limerick", "free verse", "poetic", "metaphor",
                "romantic", "epic", "acrostic", "couplet",
                "quatrain", "iambic"
            ],
            
            "email": [
                # ✅ Core email words
                "email", "mail", "letter", "message", "write", "send",
                "compose", "draft", "correspondence",
                
                # ✅ Leave-related keywords
                "leave", "sick", "vacation", "absence", "time off",
                "sick leave", "medical leave", "annual leave", "pto",
                "day off", "days off", "absent", "unavailable",
                
                # ✅ HR/Work-related
                "hr", "human resource", "human resources", 
                "manager", "boss", "supervisor", "director",
                "company", "work", "office", "workplace",
                "department", "team", "colleague", "employee",
                
                # ✅ Request types
                "request", "application", "apply", "asking", 
                "inquiry", "proposal", "permission",
                
                # ✅ Meeting/Schedule
                "meeting", "appointment", "schedule", "discuss",
                "call", "conference", "zoom", "teams",
                
                # ✅ Professional/Formal
                "professional", "formal", "business", "official",
                "corporate", "executive",
                
                # ✅ Email elements
                "dear", "sincerely", "regards", "thank", "thanks",
                "gratitude", "appreciate", "appreciation",
                
                # ✅ Email types
                "follow up", "follow-up", "following up",
                "apology", "apologize", "sorry",
                "complaint", "concern", "issue", "problem",
                "resignation", "resign", "quit", "leaving",
                "invitation", "invite", "rsvp",
                "confirmation", "confirm",
                "reminder", "reminding"
            ]
        }
        
        logger.info("TextRouter initialized with enhanced keyword matching")
    
    def calculate_expert_scores(self, prompt: str) -> Dict[str, int]:
        """Calculate scores for each expert based on keyword matching."""
        prompt_lower = prompt.lower()
        scores = {expert: 0 for expert in self.experts.keys()}
        
        # Count keyword matches for each expert
        for expert, keywords in self.expert_keywords.items():
            matches = []
            for keyword in keywords:
                if keyword in prompt_lower:
                    matches.append(keyword)
                    scores[expert] += 1
            
            if matches:
                logger.debug(f"{expert}: matched {len(matches)} keywords: {matches[:5]}")
        
        logger.info(f"Keyword match counts: {scores}")
        
        return scores
    
    def select_expert(self, prompt: str, force_expert: Optional[str] = None) -> tuple:
        """Select the best expert for the given prompt."""
        # Handle forced expert selection
        if force_expert:
            if force_expert in self.experts:
                logger.info(f"✅ Forcing expert: {force_expert}")
                return force_expert, 1.0
            else:
                raise ValueError(f"Unknown expert: {force_expert}. Available: {list(self.experts.keys())}")
        
        # Calculate scores
        scores = self.calculate_expert_scores(prompt)
        total_matches = sum(scores.values())
        
        # Handle no matches - default to email for professional requests
        if total_matches == 0:
            logger.warning("⚠️ No keyword matches found - defaulting to email expert")
            return "email", 0.5
        
        # Select expert with highest score
        best_expert = max(scores, key=scores.get)
        best_score = scores[best_expert]
        
        # Calculate confidence as percentage of total matches
        confidence = best_score / total_matches
        
        logger.info(f"✅ Selected: {best_expert.upper()} expert")
        logger.info(f"   Matches: {best_score}/{total_matches} keywords")
        logger.info(f"   Confidence: {confidence:.1%}")
        
        return best_expert, confidence
    
    def route_and_generate(
        self, 
        prompt: str, 
        max_length: int = None,
        temperature: float = None,
        force_expert: Optional[str] = None
    ) -> Dict:
        """
        Route the prompt to the appropriate expert and generate text.
        
        Args:
            prompt: User input text
            max_length: Maximum generation length (ignored for email expert)
            temperature: Sampling temperature (ignored for email expert)
            force_expert: Force a specific expert (optional)
            
        Returns:
            Dictionary containing:
            - generated_text: The generated output
            - expert: Name of expert used
            - confidence: Routing confidence score
            - all_scores: Scores for all experts
        """
        # Select the best expert
        expert_name, confidence = self.select_expert(prompt, force_expert)
        expert = self.experts[expert_name]
        
        # ✅ FIXED: Adjust parameters based on expert type
        if expert_name == "email":
            # ✅ Let email pipeline use its own optimized defaults (600, 0.5)
            # Don't override max_length or temperature!
            logger.info(f"📧 Email expert: using pipeline defaults (600, 0.5)")
            
        elif expert_name == "story":
            # Stories can be longer and more creative
            max_length = max(max_length, 200)
            temperature = max(temperature, 0.8)
            logger.info(f"📖 Story expert: adjusted max_length={max_length}, temperature={temperature}")
            
        elif expert_name == "poem":
            # Poems are typically shorter but more creative
            max_length = min(max_length, 200)
            temperature = max(temperature, 0.9)
            logger.info(f"📝 Poem expert: adjusted max_length={max_length}, temperature={temperature}")
        
        # Generate text using the selected expert
        logger.info(f"🔄 Generating with {expert_name} expert...")
        
        try:
            # ✅ FIXED: For email, don't pass parameters - use expert's defaults
            if expert_name == "email":
                generated_text = expert.generate(prompt=prompt)
            else:
                # For other experts, pass adjusted parameters
                generated_text = expert.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature
                )
            
            logger.info(f"✅ Generation successful ({len(generated_text)} characters)")
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise
        
        # Return results
        return {
            "generated_text": generated_text,
            "expert": expert_name,
            "confidence": float(confidence),
            "all_scores": self.calculate_expert_scores(prompt)
        }
    
    def get_expert_info(self) -> Dict:
        """Get information about available experts and their keywords."""
        return {
            "available_experts": list(self.experts.keys()),
            "keyword_counts": {
                expert: len(keywords) 
                for expert, keywords in self.expert_keywords.items()
            },
            "total_keywords": sum(len(kw) for kw in self.expert_keywords.values())
        }