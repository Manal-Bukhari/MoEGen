"""
Robust JSON Parser Utility
Handles malformed JSON from LLM responses with multiple fallback strategies.
"""
import json
import re
import logging

logger = logging.getLogger(__name__)


def parse_json_robust(text: str, max_attempts: int = 5) -> dict:
    """
    Parse JSON from LLM response with multiple fallback strategies.
    
    Args:
        text: Raw text that may contain JSON
        max_attempts: Maximum number of parsing attempts
        
    Returns:
        Parsed dictionary
        
    Raises:
        json.JSONDecodeError: If all parsing attempts fail
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string")
    
    # Strategy 1: Extract JSON from code blocks
    json_text = _extract_from_code_blocks(text)
    
    # Strategy 2: Find JSON object boundaries
    if '{' in json_text and '}' in json_text:
        start = json_text.find('{')
        end = json_text.rfind('}') + 1
        json_text = json_text[start:end]
    
    # Try multiple parsing strategies
    for attempt in range(max_attempts):
        try:
            # Strategy 1: Direct parse
            if attempt == 0:
                return json.loads(json_text)
            
            # Strategy 2: Fix trailing commas
            elif attempt == 1:
                cleaned = re.sub(r',(\s*[}\]])', r'\1', json_text)
                return json.loads(cleaned)
            
            # Strategy 3: Fix unescaped quotes in strings
            elif attempt == 2:
                cleaned = _fix_unescaped_quotes(json_text)
                return json.loads(cleaned)
            
            # Strategy 4: Remove comments and fix common issues
            elif attempt == 3:
                cleaned = _remove_comments(json_text)
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                cleaned = _fix_unescaped_quotes(cleaned)
                return json.loads(cleaned)
            
            # Strategy 5: Aggressive cleaning
            elif attempt == 4:
                cleaned = _aggressive_clean(json_text)
                return json.loads(cleaned)
                
        except json.JSONDecodeError as e:
            if attempt == max_attempts - 1:
                logger.error(f"❌ All JSON parsing attempts failed. Last error: {e}")
                logger.debug(f"   Problematic JSON (first 500 chars): {json_text[:500]}")
                raise
            logger.debug(f"   JSON parse attempt {attempt + 1} failed: {e}, trying next strategy...")
            continue
    
    # This should never be reached, but kept as safety fallback
    logger.error("❌ All parsing strategies exhausted without return or exception")
    raise json.JSONDecodeError("All parsing strategies failed", json_text, 0)


def _extract_from_code_blocks(text: str) -> str:
    """Extract JSON from markdown code blocks."""
    # Try ```json first
    if "```json" in text:
        parts = text.split("```json")
        if len(parts) > 1:
            json_part = parts[1].split("```")[0].strip()
            return json_part
    
    # Try ``` code blocks
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            # Find the part that looks like JSON (starts with {)
            for part in parts[1:]:
                part = part.strip()
                if part.startswith('{'):
                    return part
    
    return text.strip()


def _fix_unescaped_quotes(text: str) -> str:
    """Fix unescaped quotes inside JSON strings."""
    # This is a simplified approach - find string values and escape quotes
    # Pattern: "key": "value with "quotes" here"
    def escape_quotes_in_value(match):
        key_part = match.group(1)  # "key":
        value = match.group(2)    # value with "quotes"
        
        # Only escape quotes that are not already escaped
        value = re.sub(r'(?<!\\)"', r'\\"', value)
        return f'{key_part} "{value}"'
    
    # Match: "key": "value"
    pattern = r'("[\w_]+")\s*:\s*"([^"]*(?:"[^"]*)*)"'
    result = re.sub(pattern, escape_quotes_in_value, text)
    
    return result


def _remove_comments(text: str) -> str:
    """Remove comments from JSON-like text."""
    # Remove single-line comments (not standard JSON but sometimes present)
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


def _aggressive_clean(text: str) -> str:
    """Aggressively clean JSON text."""
    # Remove all newlines and extra whitespace
    text = ' '.join(text.split())
    
    # Fix trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix single quotes to double quotes (but be careful with apostrophes)
    # Only replace single quotes that are clearly meant to be string delimiters
    text = re.sub(r"'(\w+)':", r'"\1":', text)  # 'key': -> "key":
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)  # : 'value' -> : "value"
    
    # Remove comments
    text = _remove_comments(text)
    
    # Ensure proper JSON structure
    if not text.startswith('{'):
        # Try to find the first {
        idx = text.find('{')
        if idx >= 0:
            text = text[idx:]
    
    if not text.endswith('}'):
        # Try to find the last }
        idx = text.rfind('}')
        if idx >= 0:
            text = text[:idx+1]
    
    return text


# Note: safe_parse_json was removed as it was unused.
# If you need safe parsing with defaults, use try/except around parse_json_robust()

