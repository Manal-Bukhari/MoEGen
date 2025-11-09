from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

logger = logging.getLogger(__name__)

class BaseExpert(ABC):
    """
    Base class for all expert models.
    Each expert specializes in a specific type of text generation.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the expert with a pre-trained model.
        
        Args:
            model_name: HuggingFace model name (default: gpt2 for lightweight demo)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading {self.__class__.__name__} with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"{self.__class__.__name__} loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt that defines this expert's specialization.
        This will be prepended to user prompts to guide generation.
        """
        pass
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 150, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generate text using the expert model.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        try:
            # Prepend expert-specific system prompt
            full_prompt = self.get_system_prompt() + prompt
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                full_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the system prompt from output
            generated_text = generated_text.replace(self.get_system_prompt(), "").strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error generating text: {str(e)}"
    
    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"
