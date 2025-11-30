"""
Template Generator - Produces complete, structured email templates
"""
import logging
from typing import Dict, Any
from .base_tool import init_gemini_model, parse_json_response

logger = logging.getLogger(__name__)


class TemplateGenerator:
    """Generates structured email templates based on context and requirements."""
    
    def __init__(self, api_key: str):
        """
        Initialize TemplateGenerator.
        
        Args:
            api_key: Gemini API key (required)
        """
        self.api_key = api_key
        
        # Initialize model using shared utility
        self.model = init_gemini_model(self.api_key)
        self.enabled = self.model is not None
        
        if not self.enabled:
            logger.warning("Template Generator disabled: model initialization failed")
    
    def generate(self, extracted_context: Dict[str, Any], enhanced_query: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate structured email template.
        
        Args:
            extracted_context: Context from ContextExtractor
            enhanced_query: Optional enhanced query from router
            
        Returns:
            Dictionary with:
            - email_template: Complete structured template
            - structure: Template structure breakdown
            - sections: Individual sections (subject, greeting, body, closing)
        """
        logger.debug(f"TemplateGenerator.generate() called: intent={extracted_context.get('intent', 'N/A')}, email_type={extracted_context.get('email_type', 'N/A')}")
        
        if not self.enabled:
            logger.warning("TemplateGenerator disabled, using fallback generation")
            return self._fallback_generate(extracted_context, enhanced_query)
        
        # Build template generation prompt
        template_prompt = self._create_template_prompt(extracted_context, enhanced_query)
        
        try:
            logger.info("Calling Gemini API for template generation...")
            response = self.model.generate_content(template_prompt)
            result_text = response.text.strip()
            
            # Use shared JSON parser utility
            template_data = parse_json_response(result_text, "template generation")
            if not template_data:
                logger.warning("Falling back to fallback generation")
                return self._fallback_generate(extracted_context, enhanced_query)
            
            # Ensure all required fields exist
            result = {
                "email_template": template_data.get("email_template", ""),
                "structure": template_data.get("structure", {}),
                "sections": template_data.get("sections", {
                    "subject": "",
                    "greeting": "",
                    "body": "",
                    "closing": ""
                })
            }
            
            logger.info(f"Template generated: {len(result['email_template'])} chars")
            return result
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}", exc_info=True)
            logger.warning("Falling back to fallback generation")
            return self._fallback_generate(extracted_context, enhanced_query)
    
    def _create_template_prompt(self, extracted_context: Dict[str, Any], enhanced_query: Dict[str, Any] = None) -> str:
        """Create prompt for template generation."""
        intent = extracted_context.get("intent", "general")
        email_type = extracted_context.get("email_type", "general")
        key_entities = extracted_context.get("key_entities", {})
        
        enhanced_info = ""
        if enhanced_query:
            enhanced_info = f"""
ENHANCED QUERY:
- Email Type: {enhanced_query.get('email_type', 'unknown')}
- Tone: {enhanced_query.get('tone', 'unknown')}
- Recipient: {enhanced_query.get('recipient_type', 'unknown')}
- Key Points: {', '.join(enhanced_query.get('key_points', []))}
"""
        
        return f"""Generate a complete, structured email template based on this context:

CONTEXT:
- Intent: {intent}
- Email Type: {email_type}
- Recipient: {key_entities.get('recipient', 'general')}
- Dates: {', '.join(key_entities.get('dates', []))}
- Reason: {key_entities.get('reason', 'N/A')}
- Documentation: {key_entities.get('documentation', 'N/A')}
{enhanced_info}

Create a structured email template with proper sections. Return ONLY valid JSON:
{{
    "email_template": "Complete email template with placeholders for specific details. Include Subject, Greeting, Body paragraphs, and Closing.",
    "structure": {{
        "has_subject": true,
        "has_greeting": true,
        "has_body": true,
        "has_closing": true,
        "body_paragraphs": 2,
        "word_count_estimate": 150
    }},
    "sections": {{
        "subject": "Subject line template",
        "greeting": "Greeting template (e.g., 'Dear [Recipient],')",
        "body": "Body content template with placeholders for specific details",
        "closing": "Closing template (e.g., 'Best regards,\\n[Your Name]')"
    }}
}}

The template should be professional, complete, and ready to be filled with specific details."""
    
    def _fallback_generate(self, extracted_context: Dict[str, Any], enhanced_query: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback template generation when LLM is not available - uses generic template."""
        logger.info("Using fallback template generation")
        
        email_type = extracted_context.get("email_type", "general")
        key_entities = extracted_context.get("key_entities", {})
        recipient = key_entities.get("recipient", "Recipient")
        dates = key_entities.get("dates", [])
        
        # Generic fallback template
        subject = "Email Subject"
        if dates:
            subject += f" - {', '.join(dates)}"
        
        greeting = f"Dear {recipient},"
        body = "I am writing to [purpose].\n\n[Main content]\n\nThank you for your attention to this matter."
        closing = "\n\nBest regards,\n[Your Name]"
        
        email_template = f"Subject: {subject}\n\n{greeting}\n\n{body}{closing}"
        
        logger.info(f"Fallback template generated: {email_type} email for {recipient}")
        return {
            "email_template": email_template,
            "structure": {
                "has_subject": True,
                "has_greeting": True,
                "has_body": True,
                "has_closing": True,
                "body_paragraphs": 2,
                "word_count_estimate": len(email_template.split())
            },
            "sections": {
                "subject": subject,
                "greeting": greeting,
                "body": body,
                "closing": closing
            }
        }


