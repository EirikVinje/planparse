


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from bitsandbytes import BitsAndBytesConfig

class LLMBase:

    def __init__(
            self,
            config : dict, 
            huggingface_model : str = "NorwAI/NorwAI-Mistral-7B-instruct",
            quantization : int = 8,
            ):
        
        if quantization not in [4, 8]:
            raise ValueError("Quantization must be 4 or 8")

        self.config = config
        self.huggingface_model = huggingface_model
        self.quantization = quantization

    def load_model(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.huggingface_model, 
            padding_side="left"
        )

        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # normalized float 4 for better accuracy
        bnb_4bit_use_double_quant=True  # nested quantization for memory efficiency
        )

        # Load model in 8-bit
        model = AutoModelForCausalLM.from_pretrained(
            self.huggingface_model,
            load_in_8bit=True if self.quantization == 8 else False,
            quantization_config=quantization_config if self.quantization == 4 else None
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