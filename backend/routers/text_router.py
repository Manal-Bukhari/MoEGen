"""
Text Router - Hybrid routing system for multiple experts
"""
from typing import Dict, Optional, Any
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
    """

    def __init__(self, story_expert, poem_expert, email_expert):
        """Initialize router with expert instances."""
        # Store the instances passed from main.py
        self.experts = {
            "story": story_expert,
            "poem": poem_expert,
            "email": email_expert
        }

        # Initialize QueryEnhancer for all experts
        self.query_enhancer = QueryEnhancer()

        # Initialize LLM for routing (lazy load)
        self._llm = None
        self.use_llm_routing = True

        # High-confidence keywords for fast routing (obvious cases)
        self.high_confidence_keywords = {
            "story": [
                "write a story", "tell me a story", "create a story",
                "once upon a time", "story about", "short story",
                "write me a tale", "narrative about"
            ],
            "poem": [
                "write a poem", "compose a poem", "create a poem",
                "write poetry", "haiku about", "sonnet about",
                "poem about"
            ],
            "email": [
                "write an email", "draft an email", "compose an email",
                "email to", "write a letter", "professional email",
                "sick leave", "vacation request", "leave request"
            ]
        }

        # Comprehensive keyword lists for scoring
        self.expert_keywords = {
            "story": [
                "story", "tale", "narrative", "fiction", "novel",
                "adventure", "character", "plot", "once upon",
                "chapter", "beginning", "ending", "short story",
                "fantasy", "sci-fi", "science fiction", "mystery", "thriller",
                "hero", "villain", "protagonist", "antagonist",
                "magic", "quest", "journey", "legend", "fable",
                "chronicles", "saga", "epic", "folklore",
                "imaginary", "fictional", "storytelling", "robot", "dragon"
            ],
            "poem": [
                "poem", "poetry", "verse", "rhyme", "haiku",
                "sonnet", "stanza", "lyric", "ballad", "ode",
                "limerick", "free verse", "poetic", "metaphor",
                "romantic", "epic poem", "acrostic", "couplet",
                "quatrain", "iambic", "rhythm", "rhyming"
            ],
            "email": [
                "email", "mail", "letter", "message", "write", "send",
                "compose", "draft", "correspondence",
                "leave", "sick", "vacation", "absence", "time off",
                "sick leave", "medical leave", "annual leave", "pto",
                "day off", "days off", "absent", "unavailable",
                "hr", "human resource", "human resources",
                "manager", "boss", "supervisor", "director",
                "company", "work", "office", "workplace",
                "department", "team", "colleague", "employee",
                "request", "application", "apply", "asking",
                "inquiry", "proposal", "permission",
                "meeting", "appointment", "schedule", "discuss",
                "call", "conference", "zoom", "teams",
                "professional", "formal", "business", "official",
                "corporate", "executive",
                "dear", "sincerely", "regards", "thank", "thanks",
                "gratitude", "appreciate", "appreciation",
                "follow up", "follow-up", "following up",
                "apology", "apologize", "sorry",
                "complaint", "concern", "issue", "problem",
                "resignation", "resign", "quit", "leaving",
                "invitation", "invite", "rsvp",
                "confirmation", "confirm",
                "reminder", "reminding"
            ]
        }

        # Log active experts to verify loading
        active = [k for k, v in self.experts.items() if v is not None]
        logger.info(f"✅ Hybrid TextRouter initialized. Active experts: {active}")

    @property
    def llm(self):
        """Lazy load LLM for routing."""
        if self._llm is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if not gemini_api_key:
                    logger.warning("⚠️ GEMINI_API_KEY not set - LLM routing disabled")
                    self.use_llm_routing = False
                    return None
                
                # ✅ Use model you have access to
                self._llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-lite",  # ✅ Changed to working model
                    google_api_key=gemini_api_key,
                    temperature=0
                )
                logger.info("✅ LLM router initialized with Gemini 2.0 Flash Lite")
            except Exception as e:
                logger.error(f"❌ Failed to initialize LLM router: {e}")
                self.use_llm_routing = False
                return None
        return self._llm

    def fast_keyword_check(self, prompt: str) -> Optional[tuple]:
        """Fast pre-filter using high-confidence keywords."""
        prompt_lower = prompt.lower()
        for expert, keywords in self.high_confidence_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    # Find the context around the keyword for a more descriptive reason
                    keyword_index = prompt_lower.find(keyword)
                    context_start = max(0, keyword_index - 30)
                    context_end = min(len(prompt), keyword_index + len(keyword) + 30)
                    context = prompt[context_start:context_end].strip()
                    
                    expert_descriptions = {
                        "story": "creative narrative generation with character development and plot structure",
                        "poem": "poetic composition with various styles and verse forms",
                        "email": "professional communication with proper structure and formal tone"
                    }
                    
                    reason = f"High-confidence keyword match detected: The phrase '{keyword}' in your request ('{context}...') clearly indicates you need {expert_descriptions.get(expert, expert)}. The {expert.capitalize()} expert is specifically designed to handle this type of content."
                    logger.info(f"⚡ Fast route: '{keyword}' → {expert.upper()} expert")
                    return expert, 0.95, reason
        return None

    def llm_route(self, prompt: str) -> tuple:
        """Use LLM to intelligently route the prompt."""
        if not self.use_llm_routing or self.llm is None:
            logger.warning("⚠️ LLM routing unavailable, falling back to keyword scoring")
            return self._fallback_keyword_route(prompt)

        try:
            routing_prompt = f"""You are an expert routing system. Analyze the user's request and determine which expert should handle it.

Available experts:
- STORY: For creative narratives, tales, fiction, storytelling, adventures, fantasy, sci-fi, mystery, novels, short stories
- POEM: For poetry, verses, rhymes, haikus, sonnets, and all poetic compositions
- EMAIL: For professional emails, formal letters, business communication, leave requests, HR messages, meeting requests

User request: "{prompt}"

Respond with ONLY the expert name in uppercase: STORY, POEM, or EMAIL"""

            response = self.llm.invoke(routing_prompt)
            expert_name = response.content.strip().upper()
            expert_map = {"STORY": "story", "POEM": "poem", "EMAIL": "email"}
            selected_expert = expert_map.get(expert_name)

            if selected_expert and selected_expert in self.experts:
                # Generate a detailed reason using LLM
                reason = self._generate_llm_reason_detailed(prompt, selected_expert)
                logger.info(f"🤖 LLM route: {selected_expert.upper()} expert (confidence: 0.90)")
                return selected_expert, 0.90, reason
            else:
                logger.warning(f"⚠️ LLM returned invalid expert: {expert_name}, using fallback")
                return self._fallback_keyword_route(prompt)

        except Exception as e:
            logger.error(f"❌ LLM routing failed: {e}, using fallback")
            return self._fallback_keyword_route(prompt)
    
    def _generate_llm_reason_detailed(self, prompt: str, expert: str) -> str:
        """Generate a detailed, descriptive reason using LLM for expert selection."""
        try:
            reason_prompt = f"""Analyze why the {expert} expert is the best choice for this request.

User request: "{prompt}"

Provide a clear, specific explanation (2-3 sentences) explaining:
1. What specific elements in the request indicate {expert} expert is needed
2. Why this expert's capabilities match the request
3. What type of content the user is asking for

Be specific and reference actual words or phrases from the request if relevant.

Response (just the explanation, no labels):"""

            response = self.llm.invoke(reason_prompt)
            reason = response.content.strip()
            
            # Clean up the reason
            if reason.startswith("Explanation:"):
                reason = reason.replace("Explanation:", "").strip()
            if reason.startswith("Reason:"):
                reason = reason.replace("Reason:", "").strip()
            
            return reason
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate detailed reason: {e}, using fallback")
            return self._generate_simple_reason(prompt, expert)
    
    def _generate_simple_reason(self, prompt: str, expert: str) -> str:
        """Generate a simple keyword-based reason as fallback."""
        prompt_lower = prompt.lower()
        
        # Find specific keywords that match
        matched_keywords = []
        if expert in self.expert_keywords:
            for keyword in self.expert_keywords[expert]:
                if keyword in prompt_lower:
                    matched_keywords.append(keyword)
        
        if matched_keywords:
            top_matches = matched_keywords[:5]
            keyword_list = ', '.join([f'"{kw}"' for kw in top_matches])
            if len(matched_keywords) > 5:
                keyword_list += f" and {len(matched_keywords) - 5} more"
            
            expert_descriptions = {
                "story": "creative narrative generation with character development and plot structure",
                "poem": "poetic composition with various styles like haiku, sonnet, or free verse",
                "email": "professional communication with proper structure and formal tone"
            }
            
            return f"The request contains {expert}-related terms ({keyword_list}), indicating the need for {expert_descriptions.get(expert, expert)}. The {expert.capitalize()} expert specializes in this type of content generation."
        
        expert_descriptions = {
            "story": "creative narratives, fiction, and storytelling",
            "poem": "poetry, verses, and poetic compositions",
            "email": "professional emails and formal communication"
        }
        
        return f"Based on the request analysis, the {expert.capitalize()} expert is selected because the content requires {expert_descriptions.get(expert, expert)} capabilities."

    def _fallback_keyword_route(self, prompt: str) -> tuple:
        """Fallback routing using keyword scoring."""
        scores = self.calculate_expert_scores(prompt)
        total_matches = sum(scores.values())

        if total_matches == 0:
            reason = "No specific keywords matched. Defaulting to Email expert as it handles general text generation requests."
            logger.warning("⚠️ No keyword matches found, defaulting to email expert")
            return "email", 0.5, reason

        best_expert = max(scores, key=scores.get)
        confidence = scores[best_expert] / total_matches if total_matches > 0 else 0.5
        
        # Generate detailed reason based on keyword matches
        matched_keywords = []
        prompt_lower = prompt.lower()
        for keyword in self.expert_keywords.get(best_expert, []):
            if keyword in prompt_lower:
                matched_keywords.append(keyword)
        
        expert_descriptions = {
            "story": "creative narratives, character development, and plot structure",
            "poem": "poetry, verse composition, and poetic forms",
            "email": "professional communication, formal writing, and business correspondence"
        }
        
        if matched_keywords:
            top_matches = matched_keywords[:5]  # Show top 5 matches
            keyword_list = ', '.join([f'"{kw}"' for kw in top_matches])
            if len(matched_keywords) > 5:
                keyword_list += f", and {len(matched_keywords) - 5} more related terms"
            
            reason = f"Keyword analysis identified {scores[best_expert]} matching terms in your request, including {keyword_list}. These terms strongly indicate the need for {expert_descriptions.get(best_expert, best_expert)} capabilities, which the {best_expert.capitalize()} expert specializes in."
        else:
            reason = f"After analyzing keyword patterns, the {best_expert.capitalize()} expert was selected as it best matches the content type implied by your request. This expert handles {expert_descriptions.get(best_expert, best_expert)}."

        logger.info(f"🔍 Keyword route: {best_expert.upper()} expert (confidence: {confidence:.2f})")
        logger.debug(f"   Keyword scores: {scores}")

        return best_expert, confidence, reason

    def calculate_expert_scores(self, prompt: str) -> Dict[str, int]:
        """Calculate keyword match scores for each expert."""
        prompt_lower = prompt.lower()
        scores = {expert: 0 for expert in self.experts.keys()}

        for expert, keywords in self.expert_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    scores[expert] += 1

        return scores

    def select_expert(self, prompt: str, force_expert: Optional[str] = None) -> tuple:
        """
        Select the appropriate expert for the given prompt.

        Returns:
            tuple: (expert_name, confidence, routing_method, reason)
        """
        # Force expert if specified
        if force_expert:
            reason = f"Expert manually selected by user: {force_expert}"
            logger.info(f"🔧 Force expert requested: {force_expert}")
            if force_expert in self.experts:
                return force_expert, 1.0, "manual", reason
            else:
                raise ValueError(
                    f"Unknown expert: {force_expert}. "
                    f"Available: {list(self.experts.keys())}"
                )

        # Try fast keyword check first
        fast_result = self.fast_keyword_check(prompt)
        if fast_result:
            return fast_result[0], fast_result[1], "keyword", fast_result[2]

        # Fall back to LLM routing
        logger.info("🤖 No obvious keywords, using LLM routing...")
        expert, confidence, reason = self.llm_route(prompt)
        method = "llm" if (self.use_llm_routing and self.llm) else "fallback"
        return expert, confidence, method, reason

    def route_and_generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        force_expert: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route the prompt to appropriate expert and generate content.

        Args:
            prompt: User's input prompt
            max_length: Maximum length for generation (optional)
            temperature: Temperature for generation (optional)
            force_expert: Force a specific expert (optional)

        Returns:
            Dict with generated_text, expert, confidence, routing_method, all_scores
        """
        logger.info(f"🔄 TextRouter.route_and_generate() called")
        logger.info(f"   Prompt: {prompt[:100]}...")
        logger.debug(f"   max_length: {max_length}, temperature: {temperature}, force_expert: {force_expert}")

        # 1. Select Expert
        expert_name, confidence, routing_method, routing_reason = self.select_expert(prompt, force_expert)
        expert = self.experts[expert_name]

        # Check if expert is available
        if expert is None:
            logger.error(f"❌ Expert '{expert_name}' is not available (not loaded)")
            raise ValueError(
                f"Expert '{expert_name}' is not yet implemented or failed to load. "
                f"Available experts: {[k for k, v in self.experts.items() if v is not None]}"
            )

        logger.info(f"   ✅ Selected expert: {expert_name.upper()}")
        logger.info(f"   Confidence: {confidence:.1%}, Method: {routing_method}")

        # 2. Generate content using selected expert
        generated_text = ""
        enhanced_query = None

        try:
            if expert_name == "email":
                # Email-specific logic with query enhancement
                logger.info("📧 Invoking Email Expert...")
                enhanced_query = self.query_enhancer.enhance(prompt, expert_type="email")
                logger.debug(f"   Enhanced query keys: {list(enhanced_query.keys())}")

                generated_text = expert.generate(
                    prompt=prompt,
                    enhanced_query=enhanced_query,
                    max_length=max_length,
                    temperature=temperature
                )

            elif expert_name == "story":
                # Story-specific logic with query enhancement
                logger.info("📖 Invoking Story Expert...")
                enhanced_query = self.query_enhancer.enhance(prompt, expert_type="story")
                logger.debug(f"   Enhanced query keys: {list(enhanced_query.keys())}")

                # ✅ Pass enhanced_query to story expert
                generated_text = expert.generate(
                    prompt=prompt,
                    enhanced_query=enhanced_query,
                    max_length=max_length,
                    temperature=temperature
                )

            elif expert_name == "poem":
                # Poem-specific logic (when implemented)
                logger.info("✍️ Invoking Poem Expert...")
                enhanced_query = self.query_enhancer.enhance(prompt, expert_type="poem")

                generated_text = expert.generate(
                    prompt=prompt,
                    enhanced_query=enhanced_query,
                    max_length=max_length,
                    temperature=temperature
                )

            logger.info(f"✅ Generation successful: {len(generated_text)} chars")
            logger.debug(f"   Preview: {generated_text[:150]}...")

        except Exception as e:
            logger.error(f"❌ Generation failed with {expert_name} expert: {e}", exc_info=True)
            raise

        # 3. Return results
        return {
            "generated_text": generated_text,
            "expert": expert_name,
            "confidence": float(confidence),
            "routing_method": routing_method,
            "routing_reason": routing_reason,
            "all_scores": self.calculate_expert_scores(prompt),
            "enhanced_query": enhanced_query  # Include enhanced query for debugging/testing
        }

    def get_expert_info(self) -> Dict[str, Any]:
        """Get information about available experts."""
        return {
            "available_experts": list(self.experts.keys()),
            "active_experts": [k for k, v in self.experts.items() if v is not None],
            "keyword_counts": {
                expert: len(keywords)
                for expert, keywords in self.expert_keywords.items()
            }
        }