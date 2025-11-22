"""
Email Expert Prompts
Prompt templates for email generation.
"""
# System prompt for email generation
EMAIL_SYSTEM_PROMPT = """You are a professional email writer. Generate clear, professional, and appropriate emails based on user requests.

Your emails should be:
- Professional and appropriate in tone
- Clear and concise
- Properly formatted with subject, greeting, body, and closing
- Accurate to the user's specific requirements (dates, recipients, context)
- Free of errors and contradictions

Generate a professional email based on the user's request."""

# User prompt template
EMAIL_USER_PROMPT_TEMPLATE = """Generate a professional email with the following requirements:

{enhanced_instruction}

Email Type: {email_type}
Tone: {tone}
Recipient: {recipient_type}
Key Points: {key_points}
Special Requirements: {special_requirements}

Original request: {original_query}"""

