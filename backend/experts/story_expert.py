from experts.base_expert import BaseExpert
import logging

logger = logging.getLogger(__name__)

class StoryExpert(BaseExpert):
    """
    Expert specialized in generating creative stories, narratives, and fiction.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
        logger.info("Story Expert initialized - Ready to create narratives!")
    
    def get_system_prompt(self) -> str:
        """
        System prompt that guides the model to generate creative stories.
        """
        return """You are a creative story writer. Generate engaging, imaginative narratives with vivid descriptions and compelling characters. Focus on storytelling elements like plot, character development, and descriptive language.

Story: """
    
    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.8, **kwargs) -> str:
        """
        Generate a creative story based on the prompt.
        Uses slightly higher temperature for more creative output.
        """
        # Stories benefit from higher temperature for creativity
        if temperature < 0.7:
            temperature = 0.8
            
        # Stories need more length
        if max_length < 150:
            max_length = 200
            
        generated = super().generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            top_k=50
        )
        
        return generated
