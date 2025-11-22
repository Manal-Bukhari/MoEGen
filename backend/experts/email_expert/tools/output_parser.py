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
        
        # Ensure completeness (pass enhanced_query for sender name extraction)
        text = self._ensure_completeness(text, enhanced_query)
        
        logger.info(f"✅ Parsed to {len(text)} chars")
        return text.strip()
    
    def _remove_instruction_prefix(self, text: str) -> str:
        """Remove ALL instruction text before the actual email."""
        lines = text.split('\n')
        
        # Strategy: Find where the actual email starts by looking for multiple signals
        # The actual email should have: Subject + nearby greeting (Dear/Hello)
        
        subject_indices = []
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('subject:'):
                subject_indices.append(i)
        
        # If multiple Subject lines, the LAST one is usually the real email
        if len(subject_indices) > 1:
            # Return from the last Subject line
            return '\n'.join(lines[subject_indices[-1]:])
        elif len(subject_indices) == 1:
            idx = subject_indices[0]
            # Check if there's a greeting within next 5 lines
            has_nearby_greeting = False
            for j in range(idx, min(idx + 6, len(lines))):
                if any(greeting in lines[j].lower() for greeting in ['dear ', 'hello ', 'hi ']):
                    has_nearby_greeting = True
                    break
            
            if has_nearby_greeting:
                # This looks like the real email
                return '\n'.join(lines[idx:])
        
        # Fallback: aggressively filter instructions
        filtered = []
        skip_mode = True
        found_dear = False
        
        for line in lines:
            stripped = line.strip().lower()
            
            # Skip empty lines in skip mode
            if skip_mode and not stripped:
                continue
            
            # Skip instruction keywords
            if skip_mode:
                skip_patterns = [
                    'write a', 'write complete', 'generate', 'create a', 'requesting leave',
                    'mention that', 'keep the tone', 'must include:', 'format:',
                    'sender:', 'recipient:', 'purpose:', 'dates:', 'reason:',
                    'availability:', 'documentation:', 'closing signed'
                ]
                
                if any(pattern in stripped for pattern in skip_patterns):
                    continue
                
                # Once we find "Dear" followed by actual content, stop skipping
                if stripped.startswith('dear ') or stripped.startswith('hello '):
                    skip_mode = False
                    found_dear = True
            
            # Also check for Subject line that looks real (followed by content)
            if not found_dear and stripped.startswith('subject:') and len(stripped) > 10:
                skip_mode = False
            
            if not skip_mode:
                filtered.append(line)
        
        return '\n'.join(filtered)
        
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
    
    def _ensure_completeness(self, text: str, enhanced_query: Dict[str, Any] = None) -> str:
        """✅ Add closing if missing and replace [Your Name] with actual sender name."""
        
        # Extract sender name from enhanced_query key_points
        sender_name = None
        if enhanced_query:
            key_points = enhanced_query.get('key_points', [])
            for point in key_points:
                if 'sender:' in point.lower():
                    sender_name = point.split(':', 1)[1].strip()
                    break
        
        has_closing = bool(re.search(r'(best regards|sincerely|regards|thank you|thanks),?\s*$', 
                                     text, re.IGNORECASE | re.MULTILINE))
        
        if not has_closing:
            closing = "\n\nBest regards,\n"
            closing += sender_name if sender_name else "[Your Name]"
            text = text.rstrip() + closing
        
        # ✅ Replace [Your Name] placeholder with actual sender name
        if sender_name:
            text = re.sub(r'\[Your Name\]|\[Name\]', sender_name, text, flags=re.IGNORECASE)
        
        # If still has placeholder but no closing signature, add one
        has_signature = bool(re.search(r'\[your name\]|\[name\]', text, re.IGNORECASE))
        if has_signature and not sender_name:
            # No sender name available, keep placeholder but ensure it's properly formatted
            text = re.sub(r'\[Your Name\]|\[your name\]|\[Name\]|\[name\]', '[Your Name]', text, flags=re.IGNORECASE)
        
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