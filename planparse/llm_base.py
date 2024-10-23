


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
        

        self.tokenizer = AutoTokenizer.from_pretrained(
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
            lora_dropout=self.config["lora_config"]["lora_dropout"],
            lora_alpha=self.config["lora_config"]["lora_alpha"],
            r=self.config["lora_config"]["r"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply PEFT
        self.model = get_peft_model(model, config)

        self.model = self.model.to(self.config["device"])


    def init_training(self):
        self.model.train()
    

    def init_inference(self):
        self.model.eval()


    def generate(self):
        pass


    def parse_output(self):
        pass      