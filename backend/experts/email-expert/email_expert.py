import sys
import os
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class EmailExpert:
    """Email expert with COMPLETE content generation."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        logger.info(f"Loading: {model_name}")
        
        self.model_name = model_name
        
        # Detect device (with Apple Silicon support)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon
        else:
            self.device = "cpu"
        
        self.is_mistral = "mistral" in model_name.lower()
        self.is_phi = "phi" in model_name.lower()
        self.is_seq2seq = "t5" in model_name.lower() or "flan" in model_name.lower()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.is_seq2seq:
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            logger.info(f"✅ Seq2Seq on {self.device}")
        else:
            try:
                load_kwargs = {
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True
                }
                
                # Add eager attention for Phi-3 to avoid flash-attention issues
                if self.is_phi:
                    load_kwargs["attn_implementation"] = "eager"
                    logger.info("🔧 Using eager attention for Phi-3")
                
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                logger.info(f"✅ Loaded {model_name}")
            except RuntimeError as e:
                if "buffer size" in str(e).lower():
                    logger.info("💡 Falling back to Flan-T5")
                    from transformers import AutoModelForSeq2SeqLM
                    self.is_seq2seq = True
                    self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
                    self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
                    self.model.to(self.device)
                else:
                    raise
    
    def generate(self, enhanced_query: Dict[str, Any] = None, prompt: str = None, 
                 max_length: int = 300, temperature: float = 0.5, **kwargs) -> str:
        # ✅ Parameters are passed from EmailPipeline (already read from .env once)
        # No need to re-read .env here - just use the passed parameters
        
        if enhanced_query and isinstance(enhanced_query, dict):
            logger.info(f"📧 {enhanced_query.get('email_type', 'general')}")
            instruction = self._build_instruction(enhanced_query)
        else:
            instruction = prompt if prompt else str(enhanced_query)
        
        try:
            if self.is_seq2seq:
                return self._generate_seq2seq(instruction, max_length, temperature, **kwargs)
            else:
                return self._generate_causal(instruction, max_length, temperature, **kwargs)
        except Exception as e:
            logger.error(f"❌ {e}")
            raise
    
    def _generate_causal(self, instruction: str, max_length: int, temperature: float, **kwargs) -> str:
        logger.info(f"🔧 Starting causal generation (model: {self.model_name})")
        logger.info(f"🔧 Device: {self.device}, is_phi: {self.is_phi}, is_mistral: {self.is_mistral}")
        
        if self.is_mistral or self.is_phi:
            logger.info("🔧 Applying chat template...")
            messages = [{"role": "user", "content": instruction}]
            inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
            
            if isinstance(inputs, torch.Tensor):
                # Create attention mask for tensor input
                logger.info(f"🔧 Input tensor shape: {inputs.shape}")
                inputs_dict = {
                    'input_ids': inputs.to(self.device),
                    'attention_mask': torch.ones_like(inputs).to(self.device)
                }
                inputs = inputs_dict
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logger.info(f"🔧 Input IDs shape: {inputs['input_ids'].shape}")
        else:
            logger.info("🔧 Tokenizing without chat template...")
            tokenized = self.tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in tokenized.items()}
        
        logger.info(f"🔧 Preparing generation (max_new_tokens={max_length}, temp={temperature})...")
        
        with torch.no_grad():
            # Now inputs is always a dict with input_ids and attention_mask
            generate_kwargs = {
                **inputs, 
                "max_new_tokens": max_length,
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.85), 
                "top_k": kwargs.get("top_k", 40),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1), 
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id, 
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            # Phi-3 has cache compatibility issues - disable for now
            if self.is_phi:
                generate_kwargs["use_cache"] = False
            else:
                generate_kwargs["use_cache"] = True
            
            logger.info("🔧 Starting model.generate()...")
            import time
            start_time = time.time()
            outputs = self.model.generate(**generate_kwargs)
            elapsed = time.time() - start_time
            logger.info(f"✅ Generation completed in {elapsed:.2f}s")
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "[/INST]" in generated_text:
            response = generated_text.split("[/INST]")[-1].strip()
        elif "<|assistant|>" in generated_text:
            response = generated_text.split("<|assistant|>")[-1].strip()
        else:
            lines = generated_text.split('\n')
            result_lines = []
            email_started = False
            for line in lines:
                if email_started or line.strip().lower().startswith('subject:'):
                    email_started = True
                    result_lines.append(line)
            response = '\n'.join(result_lines) if result_lines else generated_text
        
        logger.info(f"✅ {len(response)} chars")
        return response
    
    def _generate_seq2seq(self, instruction: str, max_length: int, temperature: float, **kwargs) -> str:
        inputs = self.tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids, max_length=max_length, temperature=max(0.3, min(temperature, 0.7)),
                top_p=kwargs.get("top_p", 0.75), top_k=kwargs.get("top_k", 20),
                repetition_penalty=kwargs.get("repetition_penalty", 3.0), num_beams=1,
                do_sample=True, early_stopping=True, no_repeat_ngram_size=3
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ {len(generated_text)} chars")
        return generated_text
    
    def _build_instruction(self, enhanced_query: Dict[str, Any]) -> str:
        email_type = enhanced_query.get("email_type", "general")
        tone = enhanced_query.get("tone", "formal")
        key_points = enhanced_query.get("key_points", [])
        original_query = enhanced_query.get("original_query", "")
        
        if not self.is_seq2seq:
            instruction = f"""Write a {tone} email for: {original_query}

Type: {email_type.replace('_', ' ')}"""
            
            if key_points:
                clean_points = [p for p in key_points if not any(x in p.lower() for x in ['write', 'email', 'create'])]
                if clean_points:
                    instruction += f"\n\nInclude ALL:\n" + '\n'.join(f"- {p}" for p in clean_points)
            
            instruction += """

CRITICAL REQUIREMENTS:
- Write ONLY the complete email (no instructions, commentary, or placeholders)
- Include ALL specific details: dates, names, reasons, documentation needs
- Use 2-3 complete, detailed paragraphs
- NO fictional information or placeholders like [brackets]
- Start with "Subject:" line
- Include proper greeting and closing

Generate the complete email now:"""
        else:
            instruction = f"Write complete {tone} {email_type.replace('_', ' ')} email."
            if key_points:
                clean = [p for p in key_points if 'write' not in p.lower() and 'email' not in p.lower()]
                if clean:
                    instruction += f" Include: {', '.join(clean)}."
            instruction += " Write 2-3 complete paragraphs with all details."
        
        return instruction