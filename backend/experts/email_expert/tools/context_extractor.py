"""
Context Extractor - Intelligently extracts user intent and key information from prompts
"""
import logging
import json
import re
from typing import Dict, Any, List
from .base_tool import init_gemini_model, parse_json_response

logger = logging.getLogger(__name__)


class ContextExtractor:
    """Extracts context, intent, and key entities from user prompts intelligently."""
    
    def __init__(self, api_key: str):
        """
        Initialize ContextExtractor.
        
        Args:
            api_key: Gemini API key (required)
        """
        self.api_key = api_key
        
        # Initialize model using shared utility
        self.model = init_gemini_model(self.api_key)
        self.enabled = self.model is not None
        
        if not self.enabled:
            logger.warning("Context Extractor disabled: model initialization failed")
    
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
        logger.debug(f"ContextExtractor.extract() called with prompt length: {len(prompt)}")
        
        if not self.enabled:
            logger.warning("ContextExtractor disabled, using fallback extraction")
            return self._fallback_extract(prompt, enhanced_query)
        
        # Build extraction prompt
        extraction_prompt = self._create_extraction_prompt(prompt, enhanced_query)
        
        try:
            logger.info("Calling Gemini API for context extraction...")
            response = self.model.generate_content(extraction_prompt)
            result_text = response.text.strip()
            
            # Use shared JSON parser utility
            extracted_data = parse_json_response(result_text, "context extraction")
            if not extracted_data:
                logger.warning("Falling back to fallback extraction")
                return self._fallback_extract(prompt, enhanced_query)
            
            # Ensure all required fields exist
            result = {
                "extracted_context": extracted_data.get("extracted_context", ""),
                "intent": extracted_data.get("intent", "general"),
                "key_entities": extracted_data.get("key_entities", {}),
                "relationships": extracted_data.get("relationships", []),
                "email_type": extracted_data.get("email_type", "general"),
                "urgency": extracted_data.get("urgency", "normal"),
                "formality_level": extracted_data.get("formality_level", "professional")
            }
            
            logger.info(f"Context extracted: intent={result['intent']}, email_type={result['email_type']}, urgency={result['urgency']}")
            return result
            
        except Exception as e:
            logger.error(f"Context extraction failed: {e}", exc_info=True)
            logger.warning("Falling back to fallback extraction")
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
        """Fallback extraction when LLM is not available - uses basic pattern matching."""
        logger.info("Using fallback context extraction")
        
        # Extract dates using regex
        dates = re.findall(r'\b(\d{1,2}[-–]\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{0,4}|\d{4}-\d{2}-\d{2})\b', prompt, re.IGNORECASE)
        
        # Use enhanced_query if available, otherwise use generic defaults
        if enhanced_query:
            email_type = enhanced_query.get('email_type', 'general')
            recipient = enhanced_query.get('recipient_type', 'general')
            intent = email_type
        else:
            email_type = "general"
            recipient = "general"
            intent = "general"
        
        logger.info(f"Fallback extraction complete: intent={intent}, email_type={email_type}, recipient={recipient}")
        return {
            "extracted_context": f"User wants to write an email. {prompt}",
            "intent": intent,
            "key_entities": {
                "dates": dates,
                "recipient": recipient,
                "sender": None,
                "reason": None,
                "duration": None,
                "documentation": None,
                "deadline": None
            },
            "relationships": [
                f"email to {recipient}",
                f"dates: {', '.join(dates)}" if dates else "no specific dates"
            ],
            "email_type": email_type,
            "urgency": "normal",
            "formality_level": "professional"
        }


