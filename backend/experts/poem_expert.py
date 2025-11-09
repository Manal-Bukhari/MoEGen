from experts.base_expert import BaseExpert
import logging

logger = logging.getLogger(__name__)

class PoemExpert(BaseExpert):
    """
    Expert specialized in generating poetry, verses, and lyrical content.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
        logger.info("Poem Expert initialized - Ready to compose verses!")
    
    def get_system_prompt(self) -> str:
        """
        System prompt that guides the model to generate poetry.
        """
        return """You are a skilled poet. Create beautiful, expressive poetry with attention to rhythm, imagery, and emotional resonance. Use poetic devices like metaphor, simile, and vivid sensory language.

Poem: """
    
    def generate(self, prompt: str, max_length: int = 120, temperature: float = 0.9, **kwargs) -> str:
        """
        Generate poetry based on the prompt.
        Uses high temperature for creative, artistic output.
        Poems are typically shorter than stories.
        """
        # Poetry benefits from high temperature for creativity
        if temperature < 0.8:
            temperature = 0.9
            
        # Poems are typically shorter
        if max_length > 150:
            max_length = 120
            
        generated = super().generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            top_k=40
        )
        
        return generated
