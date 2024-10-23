


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import bitsandbytes as bnb

class LLMBase:

    def __init__(
            self,
            config : dict, 
            huggingface_model : str = "NorwAI/NorwAI-Mistral-7B-instruct",
            quantization : int = 8,
            ):
        
        self.config = config
        self.huggingface_model = huggingface_model
        self.quantization = quantization

    def load_model(self):
        

        tokenizer = AutoTokenizer.from_pretrained(
            self.huggingface_model, 
            padding_side="left"
        )
    
        # Load model in 8-bit
        model = AutoModelForCausalLM.from_pretrained(
            self.huggingface_model,
            load_in_8bit=True,
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Define LoRA config
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply PEFT
        model = get_peft_model(model, config)
        
        # Print trainable parameters info
        print_trainable_parameters(model)
