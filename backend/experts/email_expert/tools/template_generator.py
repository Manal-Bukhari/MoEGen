"""
Template Generator - Produces complete, structured email templates
"""
import os
import logging
import json
import re
from typing import Dict, Any, List
import google.generativeai as genai

logger = logging.getLogger(__name__)


class TemplateGenerator:
    """Generates structured email templates based on context and requirements."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Try multiple models for compatibility
                for model_name in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        logger.info(f"✅ Template Generator: {model_name}")
                        self.enabled = True
                        break
                    except:
                        continue
                if not hasattr(self, 'model'):
                    raise Exception("No Gemini models available")
            except Exception as e:
                logger.warning(f"⚠️ Template Generator disabled: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️ No GEMINI_API_KEY. Template Generator disabled.")
            self.enabled = False
    
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
        logger.info("📝 TemplateGenerator.generate() called")
        logger.debug(f"   Intent: {extracted_context.get('intent', 'N/A')}")
        logger.debug(f"   Email Type: {extracted_context.get('email_type', 'N/A')}")
        
        if not self.enabled:
            logger.warning("⚠️ TemplateGenerator disabled, using fallback generation")
            return self._fallback_generate(extracted_context, enhanced_query)
        
        # Build template generation prompt
        logger.debug("📝 Building template generation prompt...")
        template_prompt = self._create_template_prompt(extracted_context, enhanced_query)
        logger.debug(f"   Template prompt length: {len(template_prompt)} chars")
        
        try:
            logger.info("🤖 Calling Gemini API for template generation...")
            response = self.model.generate_content(template_prompt)
            result_text = response.text.strip()
            logger.debug(f"   Raw response length: {len(result_text)} chars")
            
            # Extract JSON from response
            logger.debug("🔧 Extracting JSON from response...")
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
                logger.debug("   Found JSON in ```json code block")
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
                logger.debug("   Found JSON in ``` code block")
            
            # Clean JSON
            logger.debug("🧹 Cleaning JSON...")
            result_text = ' '.join(result_text.split())
            result_text = re.sub(r',(\s*[}\]])', r'\1', result_text)
            result_text = result_text.replace("'", '"')
            
            if '{' in result_text and '}' in result_text:
                start = result_text.find('{')
                end = result_text.rfind('}') + 1
                result_text = result_text[start:end]
                logger.debug(f"   Extracted JSON range: {start} to {end}")
            
            try:
                logger.debug("📦 Parsing JSON...")
                template_data = json.loads(result_text)
                logger.debug(f"   Successfully parsed JSON with keys: {list(template_data.keys())}")
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parse error: {e}")
                logger.debug(f"   Problematic JSON (first 300 chars): {result_text[:300]}")
                logger.warning("⚠️ Falling back to fallback generation")
                return self._fallback_generate(extracted_context, enhanced_query)
            
            # Ensure all required fields exist
            logger.debug("🔍 Validating template data structure...")
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
            
            logger.info(f"✅ Template generated successfully:")
            logger.info(f"   Template length: {len(result['email_template'])} chars")
            logger.debug(f"   Structure keys: {list(result['structure'].keys())}")
            logger.debug(f"   Sections: {list(result['sections'].keys())}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Template generation failed: {e}", exc_info=True)
            logger.warning("⚠️ Falling back to fallback generation")
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
        """Fallback template generation when LLM is not available."""
        logger.info("🔄 Using fallback template generation")
        
        intent = extracted_context.get("intent", "general")
        email_type = extracted_context.get("email_type", "general")
        key_entities = extracted_context.get("key_entities", {})
        recipient = key_entities.get("recipient", "Recipient")
        logger.debug(f"   Intent: {intent}, Email Type: {email_type}, Recipient: {recipient}")
        
        # Generate subject based on email type
        subject_map = {
            "sick_leave": "Sick Leave Request",
            "vacation": "Vacation Leave Request",
            "meeting": "Meeting Request",
            "thank_you": "Thank You",
            "general": "Professional Correspondence"
        }
        subject = subject_map.get(email_type, "Professional Correspondence")
        
        # Add dates to subject if available
        dates = key_entities.get("dates", [])
        if dates:
            subject += f" - {', '.join(dates)}"
        
        # Generate greeting
        if recipient == "HR":
            greeting = "Dear HR Team,"
        elif recipient == "manager":
            greeting = "Dear Manager,"
        else:
            greeting = f"Dear {recipient},"
        
        # Generate body template
        if email_type == "sick_leave":
            body = f"""I am writing to request sick leave for {', '.join(dates) if dates else '[dates]'}.

I am currently unwell and will be unavailable for work during this period.

I will provide any required medical documentation upon my return to the office."""
        elif email_type == "vacation":
            body = f"""I am writing to request vacation leave for {', '.join(dates) if dates else '[dates]'}.

I would appreciate your approval for this time off.

I will ensure all my responsibilities are covered during my absence."""
        elif email_type == "meeting":
            body = f"""I would like to request a meeting to discuss [topic].

Please let me know your availability for {', '.join(dates) if dates else '[dates]'}.

I am flexible with the timing and can accommodate your schedule."""
        else:
            body = """I am writing to [purpose].

[Main content]

Thank you for your attention to this matter."""
        
        # Generate closing
        closing = "\n\nBest regards,\n[Your Name]"
        
        # Combine into full template
        email_template = f"Subject: {subject}\n\n{greeting}\n\n{body}{closing}"
        logger.debug(f"   Generated template length: {len(email_template)} chars")
        
        logger.info(f"✅ Fallback template generated: {email_type} email for {recipient}")
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


