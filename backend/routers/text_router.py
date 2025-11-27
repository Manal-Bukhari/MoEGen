# routers/text_router.py
from typing import Dict, Optional
import logging

import os
from dotenv import load_dotenv
from services.query_enhancer import QueryEnhancer

load_dotenv()

logger = logging.getLogger(__name__)

class TextRouter:
    """
    Hybrid router that uses:
    1. Fast keyword pre-filtering for obvious cases (90%+ confidence)
    2. LLM-based routing for ambiguous cases

    This approach is industry-standard and balances speed, cost, and accuracy.
    """

    def __init__(self, story_expert, poem_expert, email_expert):
        self.experts = {
            "story": story_expert,
            "poem": poem_expert,
            "email": email_expert
        }

        # Initialize QueryEnhancer for email requests
        self.query_enhancer = QueryEnhancer()

        # Initialize LLM for routing (lazy load)
        self._llm = None
        self.use_llm_routing = True  # Can be disabled for pure keyword mode

        # High-confidence keywords for fast routing (obvious cases)
        self.high_confidence_keywords = {
            "story": ["story", "tale", "narrative", "once upon", "chapter", "novel"],
            "poem": ["poem", "poetry", "haiku", "sonnet", "verse", "stanza"],
            "email": ["email", "mail", "dear", "sincerely", "regards"]
        }

        # ✅ Comprehensive keyword lists for scoring
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
        
        logger.info("✅ Hybrid TextRouter initialized (Keyword pre-filter + LLM routing)")

    @property
    def llm(self):
        """Lazy load LLM for routing"""
        if self._llm is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if not gemini_api_key:
                    logger.warning("⚠️ GEMINI_API_KEY not set - LLM routing disabled")
                    self.use_llm_routing = False
                    return None
                self._llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=gemini_api_key,
                    temperature=0
                )
                logger.info("✅ LLM router initialized with Gemini 2.5 Flash")
            except Exception as e:
                logger.error(f"❌ Failed to initialize LLM router: {e}")
                self.use_llm_routing = False
                return None
        return self._llm

    def fast_keyword_check(self, prompt: str) -> Optional[tuple]:
        """
        Fast pre-filter using high-confidence keywords.
        Returns (expert_name, confidence) if match found, else None.
        """
        prompt_lower = prompt.lower()

        for expert, keywords in self.high_confidence_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    logger.info(f"⚡ Fast route: '{keyword}' → {expert.upper()} expert")
                    return expert, 0.95  # High confidence

        return None  # No obvious match, need deeper analysis

    def llm_route(self, prompt: str) -> tuple:
        """
        Use LLM to intelligently route the prompt.
        Returns (expert_name, confidence).
        """
        if not self.use_llm_routing or self.llm is None:
            logger.warning("⚠️ LLM routing unavailable, falling back to keyword scoring")
            return self._fallback_keyword_route(prompt)

        try:
            routing_prompt = f"""You are an expert routing system. Analyze the user's request and determine which expert should handle it.

Available experts:
- STORY: For creative narratives, tales, fiction, storytelling, adventures, fantasy, sci-fi
- POEM: For poetry, verses, rhymes, haikus, sonnets, and all poetic compositions
- EMAIL: For professional emails, formal letters, business communication, leave requests, HR messages

User request: "{prompt}"

Respond with ONLY the expert name in uppercase: STORY, POEM, or EMAIL
Do not include any explanation, just the expert name."""

            response = self.llm.invoke(routing_prompt)
            expert_name = response.content.strip().upper()

            # Map to lowercase for consistency
            expert_map = {"STORY": "story", "POEM": "poem", "EMAIL": "email"}
            selected_expert = expert_map.get(expert_name)

            if selected_expert and selected_expert in self.experts:
                logger.info(f"🤖 LLM route: {selected_expert.upper()} expert")
                return selected_expert, 0.90  # LLM confidence
            else:
                logger.warning(f"⚠️ LLM returned invalid expert: {expert_name}, using fallback")
                return self._fallback_keyword_route(prompt)

        except Exception as e:
            logger.error(f"❌ LLM routing failed: {e}, using fallback")
            return self._fallback_keyword_route(prompt)

    def _fallback_keyword_route(self, prompt: str) -> tuple:
        """Fallback to keyword scoring when LLM is unavailable"""
        scores = self.calculate_expert_scores(prompt)
        total_matches = sum(scores.values())

        if total_matches == 0:
            logger.warning("⚠️ No matches - defaulting to email")
            return "email", 0.5

        best_expert = max(scores, key=scores.get)
        confidence = scores[best_expert] / total_matches
        return best_expert, confidence

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
        """
        HYBRID ROUTING: Select the best expert using multi-stage approach.

        Stage 1: Fast keyword pre-filter (obvious cases)
        Stage 2: LLM-based routing (ambiguous cases)
        Stage 3: Fallback keyword scoring

        Returns: (expert_name, confidence, routing_method)
        """
        # Handle forced expert selection
        if force_expert:
            logger.info(f"🔧 Force expert requested: {force_expert}")
            if force_expert in self.experts:
                logger.info(f"✅ Manual override: {force_expert}")
                return force_expert, 1.0, "manual"
            else:
                logger.error(f"❌ Unknown expert: {force_expert}")
                raise ValueError(f"Unknown expert: {force_expert}. Available: {list(self.experts.keys())}")

        # Stage 1: Fast keyword pre-filter
        fast_result = self.fast_keyword_check(prompt)
        if fast_result:
            expert_name, confidence = fast_result
            logger.info(f"✅ Fast route selected: {expert_name.upper()} ({confidence:.0%} confidence)")
            return expert_name, confidence, "keyword"

        # Stage 2: LLM-based routing for ambiguous cases
        logger.info("🤖 No obvious keywords, using LLM routing...")
        expert_name, confidence = self.llm_route(prompt)
        routing_method = "llm" if self.use_llm_routing and self.llm else "fallback"

        logger.info(f"✅ Selected: {expert_name.upper()} expert ({routing_method})")
        logger.info(f"   Confidence: {confidence:.1%}")

        return expert_name, confidence, routing_method
    
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
            - routing_method: How the routing was performed
            - all_scores: Scores for all experts (for debugging)
        """
        logger.info(f"🔄 TextRouter.route_and_generate() called")
        logger.info(f"   Prompt: {prompt[:100]}...")
        logger.debug(f"   max_length: {max_length}, temperature: {temperature}, force_expert: {force_expert}")
        
        # Select the best expert using hybrid routing
        logger.info("🎯 Selecting expert...")
        expert_name, confidence, routing_method = self.select_expert(prompt, force_expert)
        expert = self.experts[expert_name]
        logger.info(f"   Selected expert: {expert_name}, Confidence: {confidence:.1%}, Method: {routing_method}")
        
        # Check if expert is available
        if expert is None:
            logger.error(f"❌ Expert '{expert_name}' is not available")
            raise ValueError(f"Expert '{expert_name}' is not yet implemented. Only 'email' expert is available.")
        
        logger.debug(f"   Expert object type: {type(expert).__name__}")
        
        # Calculate all scores for debugging
        all_scores = self.calculate_expert_scores(prompt)
        
        # ✅ FIXED: Adjust parameters based on expert type
        if expert_name == "email":
            # ✅ Let email pipeline use its own optimized defaults
            # Don't override max_length or temperature!
            from experts.email_expert.config import MAX_TOKENS, TEMPERATURE
            logger.info(f"📧 Email expert: using pipeline defaults (max_tokens={MAX_TOKENS}, temperature={TEMPERATURE})")
            logger.debug(f"   Ignoring passed max_length={max_length} and temperature={temperature}")
            
        # TODO: Uncomment when story and poem experts are implemented
        # elif expert_name == "story":
        #     # Stories can be longer and more creative
        #     max_length = max(max_length, 200) if max_length else 200
        #     temperature = max(temperature, 0.8) if temperature else 0.8
        #     logger.info(f"📖 Story expert: adjusted max_length={max_length}, temperature={temperature}")
        #     
        # elif expert_name == "poem":
        #     # Poems are typically shorter but more creative
        #     max_length = min(max_length, 200) if max_length else 200
        #     temperature = max(temperature, 0.9) if temperature else 0.9
        #     logger.info(f"📝 Poem expert: adjusted max_length={max_length}, temperature={temperature}")
        
        # Generate text using the selected expert
        logger.info(f"🔄 Generating with {expert_name} expert...")
        
        try:
            # ✅ For email, enhance query first, then generate
            if expert_name == "email":
                # Enhance query using QueryEnhancer (done at router level, not in email expert)
                logger.debug("🔍 Enhancing query for email expert...")
                logger.info(f"📝 Original query: {prompt[:150]}...")
                enhanced_query = self.query_enhancer.enhance(prompt, expert_type="email")
                logger.info(f"✅ Query enhanced for email expert")
                logger.info(f"   📋 Enhanced Query Output:")
                logger.info(f"      - Email Type: {enhanced_query.get('email_type', 'N/A')}")
                logger.info(f"      - Tone: {enhanced_query.get('tone', 'N/A')}")
                logger.info(f"      - Recipient: {enhanced_query.get('recipient_type', 'N/A')}")
                logger.info(f"      - Key Points: {len(enhanced_query.get('key_points', []))} points")
                logger.info(f"      - Special Requirements: {len(enhanced_query.get('special_requirements', []))} requirements")
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
            "routing_method": routing_method,
            "all_scores": all_scores
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