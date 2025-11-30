"""
Base Tool Utilities
Shared utilities for all email expert tools.
"""
import logging
import json
from typing import Optional, Dict, Any
import google.generativeai as genai
from utils.json_parser import parse_json_robust
from ..config import MODEL_FALLBACK_LIST

logger = logging.getLogger(__name__)


def init_gemini_model(api_key: str, preferred_model: Optional[str] = None) -> Optional[genai.GenerativeModel]:
    """
    Initialize Gemini model with fallback support.
    
    Args:
        api_key: Gemini API key
        preferred_model: Preferred model name (optional)
        
    Returns:
        Initialized GenerativeModel or None if initialization fails
    """
    if not api_key:
        logger.warning("No API key provided for Gemini model initialization")
        return None
    
    try:
        genai.configure(api_key=api_key)
        
        # Try preferred model first, then fallback list
        models_to_try = []
        if preferred_model:
            models_to_try.append(preferred_model)
        models_to_try.extend(MODEL_FALLBACK_LIST)
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                logger.info(f"Initialized Gemini model: {model_name}")
                return model
            except Exception:
                continue
        
        logger.error("Failed to initialize any Gemini model")
        return None
        
    except Exception as e:
        logger.error(f"Gemini model initialization failed: {e}")
        return None


def parse_json_response(response_text: str, context: str = "response") -> Optional[Dict[str, Any]]:
    """
    Parse JSON response with robust error handling.
    
    Args:
        response_text: Text to parse as JSON
        context: Context description for error messages
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    try:
        parsed = parse_json_robust(response_text)
        logger.debug(f"Successfully parsed {context} JSON")
        return parsed
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"JSON parse error for {context}: {e}")
        logger.debug(f"Problematic JSON (first 500 chars): {response_text[:500]}")
        return None
