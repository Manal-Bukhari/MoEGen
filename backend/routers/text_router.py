# routers/text_router.py
from typing import Dict, Optional
import logging
from services.query_enhancer import QueryEnhancer

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
        # Initialize QueryEnhancer for email requests
        self.query_enhancer = QueryEnhancer()
        
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
        logger.debug(f"🎯 TextRouter.select_expert() called")
        logger.debug(f"   Prompt: {prompt[:100]}..., force_expert: {force_expert}")
        
        # Handle forced expert selection
        if force_expert:
            logger.info(f"🔧 Force expert requested: {force_expert}")
            if force_expert in self.experts:
                if self.experts[force_expert] is None:
                    logger.error(f"❌ Expert '{force_expert}' is not implemented")
                    raise ValueError(f"Expert '{force_expert}' is not yet implemented. Only 'email' expert is available.")
                logger.info(f"✅ Forcing expert: {force_expert}")
                return force_expert, 1.0
            else:
                logger.error(f"❌ Unknown expert: {force_expert}")
                raise ValueError(f"Unknown expert: {force_expert}. Available: {list(self.experts.keys())}")
        
        # Calculate scores
        logger.debug("📊 Calculating expert scores...")
        scores = self.calculate_expert_scores(prompt)
        total_matches = sum(scores.values())
        logger.debug(f"   Total keyword matches: {total_matches}")
        
        # Handle no matches - default to email for professional requests
        if total_matches == 0:
            logger.warning("⚠️ No keyword matches found - defaulting to email expert")
            if self.experts["email"] is None:
                logger.error("❌ Email expert is not available")
                raise ValueError("Email expert is not available. Cannot default to email expert.")
            return "email", 0.5
        
        # Select expert with highest score
        best_expert = max(scores, key=scores.get)
        best_score = scores[best_expert]
        
        # Calculate confidence as percentage of total matches
        confidence = best_score / total_matches
        
        logger.info(f"✅ Selected: {best_expert.upper()} expert")
        logger.info(f"   Matches: {best_score}/{total_matches} keywords")
        logger.info(f"   Confidence: {confidence:.1%}")
        
        # Verify selected expert is available
        if self.experts[best_expert] is None:
            logger.error(f"❌ Selected expert '{best_expert}' is not implemented")
            if best_expert != "email" and self.experts["email"] is not None:
                logger.warning(f"⚠️ Falling back to email expert")
                return "email", 0.5
            raise ValueError(f"Expert '{best_expert}' is not yet implemented. Only 'email' expert is available.")
        
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
        logger.info(f"🔄 TextRouter.route_and_generate() called")
        logger.info(f"   Prompt: {prompt[:100]}...")
        logger.debug(f"   max_length: {max_length}, temperature: {temperature}, force_expert: {force_expert}")
        
        # Select the best expert
        logger.info("🎯 Selecting expert...")
        expert_name, confidence = self.select_expert(prompt, force_expert)
        expert = self.experts[expert_name]
        logger.info(f"   Selected expert: {expert_name}, Confidence: {confidence:.1%}")
        
        # Check if expert is available
        if expert is None:
            logger.error(f"❌ Expert '{expert_name}' is not available")
            raise ValueError(f"Expert '{expert_name}' is not yet implemented. Only 'email' expert is available.")
        
        logger.debug(f"   Expert object type: {type(expert).__name__}")
        
        # ✅ FIXED: Adjust parameters based on expert type
        if expert_name == "email":
            # ✅ Let email pipeline use its own optimized defaults (600, 0.5)
            # Don't override max_length or temperature!
            logger.info(f"📧 Email expert: using pipeline defaults (600, 0.5)")
            logger.debug(f"   Ignoring passed max_length={max_length} and temperature={temperature}")
            
        elif expert_name == "story":
            # Stories can be longer and more creative
            max_length = max(max_length, 200) if max_length else 200
            temperature = max(temperature, 0.8) if temperature else 0.8
            logger.info(f"📖 Story expert: adjusted max_length={max_length}, temperature={temperature}")
            
        elif expert_name == "poem":
            # Poems are typically shorter but more creative
            max_length = min(max_length, 200) if max_length else 200
            temperature = max(temperature, 0.9) if temperature else 0.9
            logger.info(f"📝 Poem expert: adjusted max_length={max_length}, temperature={temperature}")
        
        # Generate text using the selected expert
        logger.info(f"🔄 Generating with {expert_name} expert...")
        
        try:
            # ✅ For email, enhance query first, then generate
            if expert_name == "email":
                # Enhance query using QueryEnhancer (done at router level, not in email expert)
                logger.debug("🔍 Enhancing query for email expert...")
                enhanced_query = self.query_enhancer.enhance(prompt, expert_type="email")
                logger.info(f"✅ Query enhanced for email expert")
                logger.debug(f"   Enhanced query keys: {list(enhanced_query.keys()) if enhanced_query else 'None'}")
                
                # Generate with enhanced query (don't pass parameters - use expert's defaults)
                logger.debug(f"📧 Calling email_expert.generate(prompt='{prompt[:50]}...', enhanced_query=...)")
                generated_text = expert.generate(
                    prompt=prompt,
                    enhanced_query=enhanced_query
                )
                logger.debug(f"   Email expert returned: {len(generated_text)} chars")
            else:
                # For other experts, pass adjusted parameters
                logger.debug(f"📝 Calling {expert_name}_expert.generate(prompt='{prompt[:50]}...', max_length={max_length}, temperature={temperature})")
                generated_text = expert.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature
                )
                logger.debug(f"   {expert_name} expert returned: {len(generated_text)} chars")
            
            logger.info(f"✅ Generation successful: {len(generated_text)} characters")
            logger.debug(f"   Generated text preview: {generated_text[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}", exc_info=True)
            raise
        
        # Return results
        logger.info(f"📤 Returning results: expert={expert_name}, confidence={confidence:.1%}, text_length={len(generated_text)}")
        result = {
            "generated_text": generated_text,
            "expert": expert_name,
            "confidence": float(confidence),
            "all_scores": self.calculate_expert_scores(prompt)
        }
        logger.debug(f"   Result keys: {list(result.keys())}")
        return result
    
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