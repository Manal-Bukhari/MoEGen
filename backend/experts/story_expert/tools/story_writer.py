"""
Story Writer - Generates actual story text (kept minimal, main generation in agent)
"""
import logging

logger = logging.getLogger(__name__)


class StoryWriter:
    """Handles story writing utilities."""
    
    def __init__(self):
        logger.info("✅ Story Writer initialized")
    
    def format_story(self, story_text: str) -> str:
        """Format story text for output."""
        
        # Remove asterisks used for Markdown bold/italics (e.g., *feeling* -> feeling)
        story_text = story_text.replace("*", "")

        # Remove excessive whitespace
        story_text = "\n".join(line.strip() for line in story_text.split("\n"))
        
        # Ensure proper paragraph spacing
        story_text = story_text.replace("\n\n\n", "\n\n")
        
        return story_text.strip()
    
    def extract_title(self, story_text: str) -> tuple:
        """Extract title if present in story."""
        lines = story_text.split("\n")
        
        # Check if first line looks like a title
        if lines and len(lines[0]) < 100 and not lines[0].endswith("."):
            title = lines[0].strip().strip("#").strip()
            body = "\n".join(lines[1:]).strip()
            return title, body
        
        return None, story_text