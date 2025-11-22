"""
Poem Expert Configuration
Configuration for Gemini-based poetry generation agent.
"""
import os
from dotenv import load_dotenv

load_dotenv()

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

