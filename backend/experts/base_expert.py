# experts/base_expert.py
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import logging

logger = logging.getLogger(__name__)

class BaseExpert(ABC):
    """Base class for all expert models."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading {self.__class__.__name__} with model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Detect model type
            if any(name in model_name.lower() for name in ['t5', 'flan', 'bart']):
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.is_seq2seq = True
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.is_seq2seq = False
            
            self.model.to(self.device)
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"{self.__class__.__name__} loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        pass
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 150, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        num_beams: int = 1,
        **kwargs
    ) -> str:
        """Generate text using the loaded model."""
        
        try:
            full_prompt = self.get_system_prompt() + prompt
            
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                if self.is_seq2seq:
                    # Seq2Seq models (Flan-T5)
                    input_length = inputs['input_ids'].shape[1]
                    max_new_tokens = max(max_length, 200)
                    
                    use_beam_search = num_beams > 1
                    
                   # In base_expert.py - Update the Seq2Seq generation section

                if self.is_seq2seq:
                    # ... existing code ...
                    
                    generation_kwargs = {
                        'input_ids': inputs['input_ids'],
                        'max_new_tokens': max_new_tokens,
                        'max_length': input_length + max_new_tokens + 100,
                        'early_stopping': True,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'attention_mask': inputs.get('attention_mask'),
                        'min_length': 50,
                        
                        # ✅ CRITICAL: Strong anti-hallucination
                        'no_repeat_ngram_size': 4,  # Changed from 3 to 4
                        'repetition_penalty': 2.5,  # Changed from 2.0 to 2.5
                        'length_penalty': 0.8,  # ✅ NEW: Slightly discourage very long outputs
                    }
                    
                    if use_beam_search:
                        generation_kwargs.update({
                            'num_beams': num_beams,
                            'num_return_sequences': 1,
                        })
                    else:
                        generation_kwargs.update({
                            'temperature': max(temperature, 0.6),  # Minimum 0.6
                            'do_sample': True,
                            'top_p': 0.9,
                            'top_k': 40,  # Reduced from 50
                        })
                    
                    outputs = self.model.generate(**generation_kwargs)
                else:
                    # Causal LM
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        max_new_tokens=max_length,
                        temperature=temperature,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        num_beams=num_beams,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        attention_mask=inputs.get('attention_mask'),
                        no_repeat_ngram_size=3  # ✅ Also for GPT-2
                    )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if not self.is_seq2seq:
                if generated_text.startswith(full_prompt):
                    generated_text = generated_text[len(full_prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"
    
    def __repr__(self):
        return self.__str__()