import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OutputParser:
    """Parses email output - MINIMAL removal, MAXIMUM preservation."""
    
    def __init__(self):
        logger.info("✅ Output Parser initialized")
    
    def parse(self, raw_output: str, enhanced_query: Dict[str, Any] = None) -> str:
        logger.info(f"🔧 Parsing {len(raw_output)} chars")
        
        text = raw_output.strip()
        
        # ONLY remove instruction prefix
        text = self._remove_instruction_prefix(text)
        
        # Fix structure if needed
        text = self._fix_email_structure(text, enhanced_query)
        
        # Fix formatting
        text = self._fix_formatting(text)
        
        # Ensure completeness
        text = self._ensure_completeness(text)
        
        logger.info(f"✅ Parsed to {len(text)} chars")
        return text.strip()
    
    def _remove_instruction_prefix(self, text: str) -> str:
        """Remove ONLY instruction before Subject line."""
        lines = text.split('\n')
        
        # Find Subject line
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('subject:'):
                return '\n'.join(lines[i:])
        
        # No subject found, remove obvious instructions at start
        filtered = []
        skip_initial = True
        
        for line in lines:
            stripped = line.strip()
            
            # Skip template lines with brackets
            if '[' in stripped and ']' in stripped and any(x in stripped.lower() for x in ['detailed', 'recipient', 'complete paragraph', 'supporting', 'your name']):
                continue
            
            if skip_initial:
                if any(stripped.lower().startswith(x) for x in [
                    'write a', 'generate', 'create a', 'task:', 'instruction:',
                    'you are', 'critical rules:', 'requirements:', 'rules:', 'critical requirements:'
                ]):
                    continue
                else:
                    skip_initial = False
            
            filtered.append(line)
        
        return '\n'.join(filtered)
    
    def _fix_email_structure(self, text: str, enhanced_query: Dict[str, Any] = None) -> str:
        """Add subject/greeting if missing."""
        has_subject = bool(re.search(r'^\s*Subject:', text, re.MULTILINE | re.IGNORECASE))
        
        if not has_subject:
            subject = "Professional Correspondence"
            if enhanced_query:
                email_type = enhanced_query.get('email_type', 'general')
                subject_map = {
                    'sick_leave': 'Sick Leave Request',
                    'vacation': 'Vacation Leave Request',
                    'meeting': 'Meeting Request',
                    'thank_you': 'Thank You',
                }
                subject = subject_map.get(email_type, 'Professional Correspondence')
            text = f"Subject: {subject}\n\n{text}"
        
        has_greeting = bool(re.search(r'^\s*Dear\s+', text, re.MULTILINE | re.IGNORECASE))
        if not has_greeting:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip().lower().startswith('subject:'):
                    lines.insert(i + 1, '')
                    lines.insert(i + 2, 'Dear Recipient,')
                    lines.insert(i + 3, '')
                    break
            text = '\n'.join(lines)
        
        return text
    
    def _fix_formatting(self, text: str) -> str:
        """Fix formatting without removing content."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        text = re.sub(r'([,.!?])(\w)', r'\1 \2', text)
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        text = re.sub(r'Subject\s*:\s*', 'Subject: ', text)
        text = re.sub(r'Dear\s+,', 'Dear Recipient,', text)
        text = re.sub(r'docx\|+', '', text)
        text = re.sub(r'\|\|+', '', text)
        return text
    
    def _ensure_completeness(self, text: str) -> str:
        """Add closing if missing."""
        has_closing = bool(re.search(r'(best regards|sincerely|regards|thank you|thanks),?\s*$', 
                                     text, re.IGNORECASE | re.MULTILINE))
        
        if not has_closing:
            text = text.rstrip() + "\n\nBest regards,\n[Your Name]"
        
        has_signature = bool(re.search(r'\[your name\]|\[name\]', text, re.IGNORECASE))
        if not has_signature and has_closing:
            text = text.rstrip() + "\n[Your Name]"
        
        return text
    
    def extract_metadata(self, email_text: str) -> Dict[str, Any]:
        subject_match = re.search(r'Subject:\s*(.+)$', email_text, re.MULTILINE)
        subject = subject_match.group(1).strip() if subject_match else None
        
        word_count = len(email_text.split())
        char_count = len(email_text)
        line_count = len(email_text.split('\n'))
        
        greeting_match = re.search(r'Dear\s+(.+?)[,\n]', email_text, re.IGNORECASE)
        greeting = greeting_match.group(0).strip() if greeting_match else None
        
        return {
            'subject': subject,
            'greeting': greeting,
            'word_count': word_count,
            'char_count': char_count,
            'line_count': line_count,
            'has_closing': bool(re.search(r'(best regards|sincerely)', email_text, re.IGNORECASE))
        }