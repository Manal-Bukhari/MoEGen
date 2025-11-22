"""
Story Expert Configuration
Configuration for Gemini-based story generation agent.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("STORY_EXPERT_MODEL", "gemini-2.0-flash-exp")

# Generation Parameters
MAX_TOKENS = int(os.getenv("STORY_MAX_TOKENS", "2000"))
TEMPERATURE = float(os.getenv("STORY_TEMPERATURE", "0.8"))
TOP_P = float(os.getenv("STORY_TOP_P", "0.95"))
TOP_K = int(os.getenv("STORY_TOP_K", "50"))

# Story-specific settings
DEFAULT_GENRE = os.getenv("STORY_DEFAULT_GENRE", "general")
DEFAULT_TONE = os.getenv("STORY_DEFAULT_TONE", "creative")

