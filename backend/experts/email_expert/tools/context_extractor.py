"""
Context Extractor - Intelligently extracts user intent and key information from prompts
"""
import os
import logging
import json
import re
from typing import Dict, Any, List
import google.generativeai as genai
from utils.json_parser import parse_json_robust

logger = logging.getLogger(__name__)


class ContextExtractor:
    """Extracts context, intent, and key entities from user prompts intelligently."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Try multiple models for compatibility
                for model_name in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        logger.info(f"✅ Context Extractor: {model_name}")
                        self.enabled = True
                        break
                    except:
                        continue
                if not hasattr(self, 'model'):
                    raise Exception("No Gemini models available")
            except Exception as e:
                logger.warning(f"⚠️ Context Extractor disabled: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️ No GEMINI_API_KEY. Context Extractor disabled.")
            self.enabled = False
    
    def extract(self, prompt: str, enhanced_query: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract context, intent, and key information from prompt.
        
        Args:
            prompt: Original user query
            enhanced_query: Optional enhanced query from router (for additional context)
            
        Returns:
            Dictionary with:
            - extracted_context: Rich context understanding
            - intent: Primary intent (sick_leave, vacation, meeting, etc.)
            - key_entities: Extracted entities (dates, recipients, reasons, etc.)
            - relationships: Relationships between entities
        """
        logger.info(f"🔍 ContextExtractor.extract() called with prompt length: {len(prompt)}")
        if enhanced_query:
            logger.debug(f"   Enhanced query provided: {list(enhanced_query.keys())}")
        
        if not self.enabled:
            logger.warning("⚠️ ContextExtractor disabled, using fallback extraction")
            return self._fallback_extract(prompt, enhanced_query)
        
        # Build extraction prompt
        logger.debug("📝 Building extraction prompt...")
        extraction_prompt = self._create_extraction_prompt(prompt, enhanced_query)
        logger.debug(f"   Extraction prompt length: {len(extraction_prompt)} chars")
        
        try:
            logger.info("🤖 Calling Gemini API for context extraction...")
            response = self.model.generate_content(extraction_prompt)
            result_text = response.text.strip()
            logger.debug(f"   Raw response length: {len(result_text)} chars")
            
            # Use robust JSON parser
            logger.debug("🔧 Parsing JSON with robust parser...")
            try:
                extracted_data = parse_json_robust(result_text)
                logger.debug(f"   Successfully parsed JSON with keys: {list(extracted_data.keys())}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"❌ JSON parse error: {e}")
                logger.debug(f"   Problematic JSON (first 500 chars): {result_text[:500]}")
                logger.warning("⚠️ Falling back to fallback extraction")
                return self._fallback_extract(prompt, enhanced_query)
            
            # Ensure all required fields exist
            logger.debug("🔍 Validating extracted data structure...")
            result = {
                "extracted_context": extracted_data.get("extracted_context", ""),
                "intent": extracted_data.get("intent", "general"),
                "key_entities": extracted_data.get("key_entities", {}),
                "relationships": extracted_data.get("relationships", []),
                "email_type": extracted_data.get("email_type", "general"),
                "urgency": extracted_data.get("urgency", "normal"),
                "formality_level": extracted_data.get("formality_level", "professional")
            }
            
            logger.info(f"✅ Context extracted successfully:")
            logger.info(f"   Intent: {result['intent']}")
            logger.info(f"   Email Type: {result['email_type']}")
            logger.info(f"   Urgency: {result['urgency']}")
            logger.info(f"   Formality: {result['formality_level']}")
            logger.debug(f"   Key entities: {list(result['key_entities'].keys())}")
            logger.debug(f"   Relationships count: {len(result['relationships'])}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Context extraction failed: {e}", exc_info=True)
            logger.warning("⚠️ Falling back to fallback extraction")
            return self._fallback_extract(prompt, enhanced_query)
    
    def _create_extraction_prompt(self, prompt: str, enhanced_query: Dict[str, Any] = None) -> str:
        """Create prompt for context extraction."""
        enhanced_info = ""
        if enhanced_query:
            enhanced_info = f"""
ENHANCED QUERY INFO:
- Email Type: {enhanced_query.get('email_type', 'unknown')}
- Tone: {enhanced_query.get('tone', 'unknown')}
- Recipient: {enhanced_query.get('recipient_type', 'unknown')}
- Key Points: {', '.join(enhanced_query.get('key_points', []))}
"""
        
        return f"""Analyze this email request and extract intelligent context:

USER REQUEST: {prompt}
{enhanced_info}

Extract and return ONLY valid JSON with:
{{
    "extracted_context": "Rich understanding of what the user wants - summarize the full context, purpose, and requirements",
    "intent": "Primary intent (sick_leave, vacation, meeting, thank_you, inquiry, complaint, etc.)",
    "email_type": "Type of email (sick_leave, vacation, meeting, thank_you, general, etc.)",
    "key_entities": {{
        "dates": ["extracted dates if any"],
        "recipient": "who the email is for (HR, manager, client, etc.)",
        "sender": "sender name if mentioned",
        "reason": "reason for email (illness, vacation, meeting request, etc.)",
        "duration": "duration if applicable",
        "documentation": "required documentation if mentioned",
        "deadline": "deadline if mentioned"
    }},
    "relationships": [
        "relationships between entities, e.g., 'sick leave requested for specific dates', 'email to HR about leave'"
    ],
    "urgency": "urgent/normal/low",
    "formality_level": "formal/semi-formal/casual/professional"
}}

Be thorough and extract all relevant information."""
    
    def _fallback_extract(self, prompt: str, enhanced_query: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback extraction when LLM is not available."""
        logger.info("🔄 Using fallback context extraction")
        logger.debug(f"   Prompt: {prompt[:100]}...")
        
        prompt_lower = prompt.lower()
        
        # Extract dates
        logger.debug("📅 Extracting dates from prompt...")
        dates = re.findall(r'\b(\d{1,2}[-–]\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{0,4}|\d{4}-\d{2}-\d{2})\b', prompt, re.IGNORECASE)
        if dates:
            logger.debug(f"   Found dates: {dates}")
        
        # Determine intent
        logger.debug("🎯 Determining intent from prompt keywords...")
        intent = "general"
        email_type = "general"
        if any(word in prompt_lower for word in ["sick", "ill", "unwell", "medical"]):
            intent = "sick_leave"
            email_type = "sick_leave"
        elif any(word in prompt_lower for word in ["vacation", "holiday", "time off", "leave"]):
            intent = "vacation"
            email_type = "vacation"
        elif any(word in prompt_lower for word in ["meeting", "schedule", "appointment"]):
            intent = "meeting"
            email_type = "meeting"
        elif any(word in prompt_lower for word in ["thank", "appreciate", "grateful"]):
            intent = "thank_you"
            email_type = "thank_you"
        
        # Extract recipient
        recipient = "general"
        if "hr" in prompt_lower or "human resource" in prompt_lower:
            recipient = "HR"
        elif "manager" in prompt_lower or "supervisor" in prompt_lower:
            recipient = "manager"
        elif "client" in prompt_lower:
            recipient = "client"
        
        # Extract reason
        reason = ""
        if "unwell" in prompt_lower or "sick" in prompt_lower:
            reason = "illness"
        elif "vacation" in prompt_lower:
            reason = "vacation"
        
        # Use enhanced_query if available
        if enhanced_query:
            logger.debug("📝 Using enhanced_query to override values...")
            email_type = enhanced_query.get('email_type', email_type)
            recipient = enhanced_query.get('recipient_type', recipient)
        
        logger.info(f"✅ Fallback extraction complete: intent={intent}, email_type={email_type}, recipient={recipient}")
        return {
            "extracted_context": f"User wants to write a {email_type} email. {prompt}",
            "intent": intent,
            "key_entities": {
                "dates": dates,
                "recipient": recipient,
                "sender": None,
                "reason": reason,
                "duration": None,
                "documentation": "medical documentation" if intent == "sick_leave" else None,
                "deadline": None
            },
            "relationships": [
                f"{intent} email to {recipient}",
                f"dates: {', '.join(dates)}" if dates else "no specific dates"
            ],
            "email_type": email_type,
            "urgency": "normal",
            "formality_level": "professional"
        }


