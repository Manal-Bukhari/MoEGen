"""
Email Expert Configuration
Centralized configuration for Gemini-based email generation agent.
All .env configuration is read here and exported to other modules.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("EMAIL_EXPERT_MODEL", "gemini-2.5-flash")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gemini-2.5-flash")

# Model fallback list for tools (in order of preference)
MODEL_FALLBACK_LIST = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

# Generation Parameters
MAX_TOKENS = int(os.getenv("EMAIL_MAX_TOKENS", "2000"))
TEMPERATURE = float(os.getenv("EMAIL_TEMPERATURE", "0.5"))

# Email-specific settings
USE_EVALUATOR = os.getenv("USE_EMAIL_EVALUATOR", "true").lower() == "true"
EVALUATOR_THRESHOLD = float(os.getenv("EVALUATOR_THRESHOLD", "7.0"))
EVALUATOR_MAX_RETRIES = int(os.getenv("EVALUATOR_MAX_RETRIES", "2"))

# Evaluator scoring parameters
EVALUATOR_PENALTY_PER_ISSUE = float(os.getenv("EVALUATOR_PENALTY_PER_ISSUE", "2.5"))
EVALUATOR_MAX_PENALTY = float(os.getenv("EVALUATOR_MAX_PENALTY", "8.0"))
EVALUATOR_CRITICAL_ISSUE_SCORE_CAP = float(os.getenv("EVALUATOR_CRITICAL_ISSUE_SCORE_CAP", "5.0"))
EVALUATOR_DEFAULT_SCORE = float(os.getenv("EVALUATOR_DEFAULT_SCORE", "10.0"))
EVALUATOR_MAX_CRITICAL_ERRORS_DISPLAY = int(os.getenv("EVALUATOR_MAX_CRITICAL_ERRORS_DISPLAY", "5"))

# Email generation retry parameters
MAX_TOKENS_RETRY_MULTIPLIER = float(os.getenv("MAX_TOKENS_RETRY_MULTIPLIER", "2.0"))
TEMPERATURE_INCREMENT_PER_ATTEMPT = float(os.getenv("TEMPERATURE_INCREMENT_PER_ATTEMPT", "0.1"))
MAX_TEMPERATURE = float(os.getenv("MAX_TEMPERATURE", "1.0"))
MAX_TOKENS_RETRY_MAX_ATTEMPTS = int(os.getenv("MAX_TOKENS_RETRY_MAX_ATTEMPTS", "1"))

# Email validation parameters
MIN_EMAIL_LENGTH = int(os.getenv("MIN_EMAIL_LENGTH", "50"))
MIN_BODY_LENGTH = int(os.getenv("MIN_BODY_LENGTH", "100"))

