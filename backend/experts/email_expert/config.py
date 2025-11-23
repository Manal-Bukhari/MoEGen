"""
Email Expert Configuration
Configuration for Gemini-based email generation agent.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("EMAIL_EXPERT_MODEL", "gemini-2.5-flash")

# Generation Parameters
MAX_TOKENS = int(os.getenv("EMAIL_MAX_TOKENS", "2000"))  # Increased from 600 to prevent truncation
TEMPERATURE = float(os.getenv("EMAIL_TEMPERATURE", "0.5"))
#
# Email-specific settings
USE_EVALUATOR = os.getenv("USE_EMAIL_EVALUATOR", "true").lower() == "true"
EVALUATOR_THRESHOLD = float(os.getenv("EVALUATOR_THRESHOLD", "7.0"))
EVALUATOR_MAX_RETRIES = int(os.getenv("EVALUATOR_MAX_RETRIES", "2"))

