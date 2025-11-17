#!/usr/bin/env python3
"""
Email Generation Pipeline - Environment-Configured
- All parameters configurable via .env file
- Better defaults (600 tokens, temp 0.5) 
- More logging
- Preserves all content
"""

import sys
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from query_enhancer import QueryEnhancer
from email_expert import EmailExpert
from output_parser import OutputParser
from email_evaluator import EmailEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailPipeline:
    """Orchestrates email generation with FIXED parameters."""
    
    def __init__(
        self,
        google_api_key: str = None,
        model_name: str = None,
        use_evaluator: bool = None,
        eval_threshold: float = None,
        max_retries: int = None
    ):
        """
        Initialize pipeline.
        All parameters can be overridden or will use .env defaults.
        
        Recommended models for Mac:
        - "microsoft/Phi-3-mini-4k-instruct" (3.8B, ~8GB RAM, high quality)
        - "google/flan-t5-large" (780MB, reliable fallback)
        - "mistralai/Mistral-7B-Instruct-v0.2" (7B, requires 16GB+ RAM)
        """
        # ✅ Read from environment ONCE and store as instance variables
        model_name = model_name or os.getenv("EMAIL_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
        use_evaluator = use_evaluator if use_evaluator is not None else os.getenv("USE_EMAIL_EVALUATOR", "true").lower() == "true"
        eval_threshold = eval_threshold if eval_threshold is not None else float(os.getenv("EVALUATOR_THRESHOLD", "7.0"))
        max_retries = max_retries if max_retries is not None else int(os.getenv("EVALUATOR_MAX_RETRIES", "2"))
        
        # ✅ Store generation parameters as instance variables (read once from .env)
        self.default_max_length = int(os.getenv("EMAIL_MAX_LENGTH", "300"))
        self.default_temperature = float(os.getenv("EMAIL_TEMPERATURE", "0.5"))
        
        logger.info("🚀 Initializing Email Pipeline (Environment-Configured)")
        logger.info(f"📧 Model: {model_name}")
        logger.info(f"📊 Max Length: {self.default_max_length}")
        logger.info(f"🌡️  Temperature: {self.default_temperature}")
        
        self.query_enhancer = QueryEnhancer(api_key=google_api_key)
        self.email_expert = EmailExpert(model_name=model_name)
        self.output_parser = OutputParser()
        
        # Initialize evaluator
        self.use_evaluator = use_evaluator
        if use_evaluator:
            self.evaluator = EmailEvaluator(
                api_key=google_api_key,
                threshold=eval_threshold,
                max_retries=max_retries
            )
            if self.evaluator.enabled:
                logger.info(f"✅ Evaluator enabled (threshold: {eval_threshold})")
            else:
                logger.info("⚠️ Evaluator disabled (no API key)")
                self.use_evaluator = False
        else:
            logger.info("⚠️ Evaluator disabled")
        
        logger.info("✅ Pipeline ready")
    
    def generate_email(
        self,
        user_query: str,
        max_length: int = None,
        temperature: float = None
    ) -> dict:
        # ✅ Use instance variables (already read from .env once in __init__)
        max_length = max_length if max_length is not None else self.default_max_length
        temperature = temperature if temperature is not None else self.default_temperature
        
        logger.info(f"📝 Query: {user_query[:100]}...")
        logger.info(f"⚙️  Using max_length={max_length}, temperature={temperature}")
        
        try:
            # Step 1: Enhance
            logger.info("Step 1/3: Enhancing query...")
            enhanced_query = self.query_enhancer.enhance(user_query)
            logger.info(f"  Type: {enhanced_query.get('email_type')}, Tone: {enhanced_query.get('tone')}")
            
            # Step 2: Generate
            logger.info(f"Step 2/3: Generating (max_length={max_length}, temp={temperature})...")
            raw_output = self.email_expert.generate(
                enhanced_query=enhanced_query,
                max_length=max_length,
                temperature=temperature
            )
            logger.info(f"  Generated: {len(raw_output)} chars - {raw_output[:100]}...")
            
            # Step 3: Parse
            logger.info("Step 3/3: Parsing...")
            final_email = self.output_parser.parse(raw_output, enhanced_query)
            
            # Step 4: Evaluate and regenerate if needed
            evaluation = None
            if self.use_evaluator and self.evaluator.enabled:
                logger.info("Step 4/4: Evaluating...")
                final_email, evaluation = self.evaluator.evaluate_with_regeneration(
                    prompt=user_query,
                    generated_email=final_email,
                    email_expert=self.email_expert,
                    enhanced_query=enhanced_query,
                    max_length=max_length,  # ✅ Pass the configured max_length
                    temperature=temperature  # ✅ Pass the configured temperature
                )
                logger.info(f"  Final score: {evaluation['score']:.1f}")
            else:
                logger.info("Step 4/4: Evaluation skipped")
            
            metadata = self.output_parser.extract_metadata(final_email)
            
            logger.info(f"✅ Complete! {len(final_email)} chars, {metadata['word_count']} words")
            
            return {
                'final_email': final_email,
                'enhanced_query': enhanced_query,
                'raw_output': raw_output,
                'metadata': metadata,
                'evaluation': evaluation
            }
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate(self, prompt: str, max_length: int = None, temperature: float = None, **kwargs) -> str:
        """Simple interface for router. ✅ FIXED: No hardcoded defaults - reads from .env"""
        result = self.generate_email(prompt, max_length, temperature)
        return result['final_email']


if __name__ == "__main__":
    pipeline = EmailPipeline()
    result = pipeline.generate_email(
        "Write a professional sick-leave email to HR requesting leave for 17-18 November 2025"
    )
    logger.info(f"Generated email:\n{result['final_email']}")