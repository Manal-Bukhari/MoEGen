"""
Simple test for free tier - ONLY 1 LLM CALL
"""
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_story_simple():
    """Test story generation with minimal LLM calls."""
    from experts.story_expert.agent import StoryExpertAgent
    
    logger.info("="*60)
    logger.info("🆓 FREE TIER TEST - Story Expert (1 LLM call only)")
    logger.info("="*60)
    
    # Check .env settings
    from experts.story_expert import config
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Model: {config.GEMINI_MODEL}")
    logger.info(f"   Context Extractor: {config.USE_CONTEXT_EXTRACTOR}")
    logger.info(f"   Story Planner: {config.USE_STORY_PLANNER}")
    logger.info(f"   Character Generator: {config.USE_CHARACTER_GENERATOR}")
    logger.info(f"   Evaluator: {config.USE_EVALUATOR}")
    
    if any([config.USE_CONTEXT_EXTRACTOR, config.USE_STORY_PLANNER, 
            config.USE_CHARACTER_GENERATOR, config.USE_EVALUATOR]):
        logger.warning("\n⚠️  WARNING: Some expensive tools are enabled!")
        logger.warning("   This will use multiple LLM calls and may hit rate limits")
        logger.warning("   Set all to 'false' in .env for free tier")
    else:
        logger.info("\n✅ All expensive tools disabled - will use only 1 LLM call!")
    
    agent = StoryExpertAgent()
    
    prompt = "Write a short story about a lonely robot who finds a flower"
    
    logger.info(f"\n📝 Prompt: {prompt}")
    logger.info("\n🚀 Generating story...\n")
    
    try:
        story = agent.generate(prompt)
        
        print("\n" + "="*60)
        print("✅ GENERATED STORY:")
        print("="*60)
        print(story)
        print("="*60)
        
        # ✅ FIX: Calculate paragraph count before f-string
        paragraph_count = len([p for p in story.split('\n\n') if p.strip()])
        
        print(f"\n📊 Stats:")
        print(f"   Length: {len(story)} characters")
        print(f"   Words: {len(story.split())} words")
        print(f"   Paragraphs: {paragraph_count}")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        logger.info("\n💡 Tips:")
        logger.info("   1. Check your .env file has correct model name")
        logger.info("   2. Make sure all expensive tools are disabled")
        logger.info("   3. Wait 60 seconds if you just hit rate limit")
        logger.info("   4. Check your API key is valid")

if __name__ == "__main__":
    test_story_simple()