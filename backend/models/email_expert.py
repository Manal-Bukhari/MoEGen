from experts.base_expert import BaseExpert
import logging

logger = logging.getLogger(__name__)

class EmailExpert(BaseExpert):
    """
    Expert specialized in generating professional emails and formal communications.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
        logger.info("Email Expert initialized - Ready to draft professional communications!")
    
    def get_system_prompt(self) -> str:
        """
        System prompt that guides the model to generate professional emails.
        """
        return """You are a professional email writer. Create clear, concise, and courteous business communications. Use formal language, proper structure, and maintain a professional tone throughout.

Email: """
    
    def generate(self, prompt: str, max_length: int = 150, temperature: float = 0.6, **kwargs) -> str:
        """
        Generate a professional email based on the prompt.
        Uses lower temperature for more focused, professional output.
        """
        # Emails benefit from lower temperature for clarity and professionalism
        if temperature > 0.7:
            temperature = 0.6
            
        # Emails are typically medium length
        if max_length < 100:
            max_length = 150
            
        generated = super().generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            top_k=50
        )
        
        return generated
