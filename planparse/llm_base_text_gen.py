import os


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from bitsandbytes import BitsAndBytesConfig

class LLMBaseTextGen:

    def __init__(
            self, 
            config : dict,
            ):
        
        self.config = config
  

    def load_model(self):
        
        if os.path.pardir(self.config["savepath"]) != "models":

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["huggingface_model"], 
                padding_side="left"
            )

            assert not self.config["load_in_4bit"] or not self.config["load_in_8bit"], "Must load model in 4-bit or 8-bit. Not both."

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config["load_in_4bit"],
                load_in_8bit=self.config["load_in_8bit"],
            )

            torch_dtype = torch.float16 if self.config["torch_dtype"] == "float16" else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                self.config["huggingface_model"],
                quantization_config=quantization_config,
                torch_dtype=torch_dtype
            )
            
            model = prepare_model_for_kbit_training(model)
            
            config = LoraConfig(
                lora_dropout=self.config["lora_dropout"],
                lora_alpha=self.config["lora_alpha"],
                r=self.config["r"],
                task_type="CAUSAL_LM",
                bias="none",
            )
        
            self.model = get_peft_model(model, config)

        else:

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["savepath"], 
                padding_side="left",
                )
            
            torch_dtype = torch.float16 if self.config["torch_dtype"] == "float16" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["savepath"],
                torch_dtype=torch_dtype,
                )


    def init_training(self):
        self.model.train()
    

    def to_device(self, device : str = None):

        if device is None:
            device = self.config["device"]

        self.model = self.model.to(device)


    def init_inference(self):
        self.model.eval()


    def generate(self, X):
        pass

    def parse_output(self):
        pass


    def save(self):

        self.model.save_pretrained(self.config["savepath"])