from typing import List, Dict, Any
import logging
import os


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from json_repair import repair_json
from huggingface_hub import login
from tqdm import tqdm
import numpy as np
import torch

logger = logging.getLogger("planparse")

class LLMBaseTextGen:

    def __init__(
            self, 
            config : dict,
            local_path : str = None,
            ):
        

        self.model_config = config["model_config"]
        self.generation_config = config["generation_config"]

        if self.model_config["access_token"] is not None:
            with open(self.model_config["access_token"], "r") as f:
                self.access_token = f.read().strip()

            login(token=self.access_token)

        self.local_path = local_path

        
    def save(self, path : str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


    def init_training(self):
        self.model.train()
    

    def to_device(self, device : str = None):
        if device is None:
            device = self.model_config["device"]
        self.model = self.model.to(device)


    def init_inference(self):
        self.model.eval()


    def load_model(self):
        
        if self.local_path is None:

            low_cpu_mem_usage = True if self.model_config["load_in_8bit"] or self.model_config["load_in_4bit"] else False
            torch_dtype = torch.float16 if self.model_config["torch_dtype"] == "float16" else torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["huggingface_model"], 
                padding_side="left",
            )
            self.tokenizer.add_special_tokens({'pad_token': "[PAD]"})
        
            assert not self.model_config["load_in_4bit"] or not self.model_config["load_in_8bit"], "Must load model in 4-bit or 8-bit. Not both."

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.model_config["load_in_4bit"],
                load_in_8bit=self.model_config["load_in_8bit"],
            )

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_config["huggingface_model"],
                quantization_config=quantization_config,
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch_dtype,
            )
            
            if self.model_config["load_in_4bit"] or self.model_config["load_in_8bit"]:
                model = prepare_model_for_kbit_training(model)
            
            model_config = LoraConfig(
                lora_dropout=self.model_config["lora_dropout"],
                lora_alpha=self.model_config["lora_alpha"],
                r=self.model_config["r"],
                task_type="CAUSAL_LM",
                bias="none",
            )
        
            self.model = get_peft_model(model, model_config)

        else:

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_path, 
                padding_side="left",
                )
            
            torch_dtype = torch.float16 if self.model_config["torch_dtype"] == "float16" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_path,
                )

    
    def generate(self, prompts : List, generation_config : Dict) :
        
        fixed_outputs = []
        
        max_new_tokens = self.generation_config["max_new_tokens"]

        for doc in tqdm(prompts, desc="generating outputs", disable=False):

            try:
                if self.generation_config["return_token_type_ids"] == False:
                    inputs = self.tokenizer(
                        return_tensors="pt", 
                        padding="longest", 
                        truncation=True,
                        text=[doc],
                        return_token_type_ids=False
                        )
            except:
                inputs = self.tokenizer(
                    return_tensors="pt", 
                    padding="longest", 
                    truncation=True,
                    text=[doc],
                    )
            
            print(inputs["input_ids"].size(1)+max_new_tokens)

            if "cuda" in str(self.model.device):
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                
                outputs = self.model.generate(
                    **inputs, 
                    #max_length=inputs["input_ids"].size(1) + max_new_tokens,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=self.generation_config["do_sample"],
                    )

                outputs = outputs[:, inputs["input_ids"].size(1):]
        
            decoded_outputs = self.tokenizer.batch_decode(
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
                sequences=outputs, 
                )
        
            # decoded_outputs = [self._parse_outputs(decoded_output) for decoded_output in decoded_outputs]

            fixed_outputs.extend(decoded_outputs)
            
            if "cuda" in str(self.model.device):
                torch.cuda.empty_cache()

        return fixed_outputs
            
            
    def _parse_outputs(self, decoded_output):
        
        parsed_outputs = repair_json(decoded_output, return_objects=True)
        
        if isinstance(parsed_outputs, list):
            parsed_outputs = parsed_outputs[-1]
        
        return parsed_outputs

    