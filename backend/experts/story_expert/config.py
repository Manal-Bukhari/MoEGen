"""
Story Expert Configuration
Configuration for Gemini-based story generation agent.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("STORY_EXPERT_MODEL", "gemini-1.5-flash")  # ✅ FIXED

# Generation Parameters
MAX_TOKENS = int(os.getenv("STORY_MAX_TOKENS", "4000"))
TEMPERATURE = float(os.getenv("STORY_TEMPERATURE", "0.8"))
TOP_P = float(os.getenv("STORY_TOP_P", "0.95"))
TOP_K = int(os.getenv("STORY_TOP_K", "50"))

# Story-specific settings
DEFAULT_GENRE = os.getenv("STORY_DEFAULT_GENRE", "general")
DEFAULT_TONE = os.getenv("STORY_DEFAULT_TONE", "creative")

# 🔥 Feature flags to reduce LLM calls
USE_CONTEXT_EXTRACTOR = os.getenv("USE_STORY_CONTEXT_EXTRACTOR", "false").lower() == "true"
USE_STORY_PLANNER = os.getenv("USE_STORY_PLANNER", "false").lower() == "true"
USE_CHARACTER_GENERATOR = os.getenv("USE_CHARACTER_GENERATOR", "false").lower() == "true"
USE_EVALUATOR = os.getenv("USE_STORY_EVALUATOR", "false").lower() == "true"

EVALUATOR_THRESHOLD = float(os.getenv("STORY_EVALUATOR_THRESHOLD", "7.0"))
EVALUATOR_MAX_RETRIES = int(os.getenv("STORY_EVALUATOR_MAX_RETRIES", "0"))

# Validate configuration
def validate_config():
    """Validate required configuration."""
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY is required!")
        raise ValueError("GEMINI_API_KEY must be set in environment variables")
    
    if MAX_TOKENS < 1000:
        logger.warning(f"⚠️ MAX_TOKENS is very low ({MAX_TOKENS}), stories may be truncated")
    
    logger.info(f"✅ Config validated: Model={GEMINI_MODEL}, MaxTokens={MAX_TOKENS}")

# Validate on import
try:
    validate_config()
except ValueError as e:
    logger.warning(f"⚠️ Config validation failed: {e}")