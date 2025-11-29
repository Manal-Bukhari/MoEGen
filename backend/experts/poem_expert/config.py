"""
Poem Expert Configuration
Configuration for Gemini-based poetry generation agent.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("POEM_EXPERT_MODEL", "gemini-2.0-flash-exp")

# Generation Parameters
MAX_TOKENS = int(os.getenv("POEM_MAX_TOKENS", "1000"))
TEMPERATURE = float(os.getenv("POEM_TEMPERATURE", "0.9"))
TOP_P = float(os.getenv("POEM_TOP_P", "0.95"))
TOP_K = int(os.getenv("POEM_TOP_K", "40"))

# Poetry-specific settings
DEFAULT_POEM_TYPE = os.getenv("POEM_DEFAULT_TYPE", "free_verse")
DEFAULT_TONE = os.getenv("POEM_DEFAULT_TONE", "expressive")

# Evaluator Settings (OPTIONAL - disable to save API costs)
USE_EVALUATOR = os.getenv("USE_POEM_EVALUATOR", "false").lower() == "true"
EVALUATOR_THRESHOLD = float(os.getenv("POEM_EVALUATOR_THRESHOLD", "7.0"))
EVALUATOR_MAX_RETRIES = int(os.getenv("POEM_EVALUATOR_MAX_RETRIES", "1"))

# Validation
def validate_config():
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY required!")
        raise ValueError("GEMINI_API_KEY must be set")
    logger.info(f"✅ Config validated: {GEMINI_MODEL}")

try:
    validate_config()
except ValueError as e:
    logger.warning(f"⚠️ Config validation failed: {e}")
