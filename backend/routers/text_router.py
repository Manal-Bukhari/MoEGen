# routers/text_router.py
from typing import Dict, Optional
import logging

import os
from dotenv import load_dotenv

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
        # Select the best expert using hybrid routing
        expert_name, confidence, routing_method = self.select_expert(prompt, force_expert)
        expert = self.experts[expert_name]

        # Calculate all scores for debugging
        all_scores = self.calculate_expert_scores(prompt)

        # ✅ TESTING MODE: Just return routing info instead of generating
        logger.info(f"🎯 HYBRID ROUTING TEST MODE")
        logger.info(f"📍 Selected Expert: {expert_name.upper()}")
        logger.info(f"🔀 Routing Method: {routing_method}")
        logger.info(f"📊 Confidence: {confidence:.1%}")

        # Routing method emoji
        method_emoji = {
            "keyword": "⚡",
            "llm": "🤖",
            "fallback": "🔄",
            "manual": "👤"
        }

        # Create a detailed test response
        test_response = f"""
🎯 HYBRID ROUTING DECISION
==========================
✅ Selected Expert: {expert_name.upper()}
{method_emoji.get(routing_method, '🔀')} Routing Method: {routing_method.upper()}
📊 Confidence Score: {confidence:.1%}

📈 Keyword Match Analysis:
- Story Expert: {all_scores.get('story', 0)} matches
- Poem Expert: {all_scores.get('poem', 0)} matches
- Email Expert: {all_scores.get('email', 0)} matches

📝 Your Prompt: "{prompt}"

💡 Routing Strategy Used:
{self._get_routing_explanation(routing_method)}

🔧 This is a TEST RESPONSE - agents are not yet implemented.
Once agents are ready, this will be replaced with actual generation.
"""

        # Return test results
        return {
            "generated_text": test_response,
            "expert": expert_name,
            "confidence": float(confidence),
            "routing_method": routing_method,
            "all_scores": all_scores
        }

    def _get_routing_explanation(self, method: str) -> str:
        """Get explanation for routing method used"""
        explanations = {
            "keyword": "⚡ Fast keyword pre-filter detected obvious match (e.g., 'poem', 'story', 'email')",
            "llm": "🤖 LLM analyzed the prompt semantically for intelligent routing",
            "fallback": "🔄 Keyword scoring used (LLM unavailable or no API key)",
            "manual": "👤 User manually selected this expert"
        }
        return explanations.get(method, "Unknown routing method")
    
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