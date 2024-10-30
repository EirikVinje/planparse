from typing import List, Dict, Any
import os


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from json_repair import repair_json
from tqdm import tqdm
import torch

class LLMBaseTextGen:

    def __init__(
            self, 
            config : dict,
            ):
        
        """
        Functions:

        - load_model()
        - init_inference()
        - init_training()
        - generate()
        - parse_output()
        - save()
        
        """

        self.config = config
    

    def save(self):
        self.model.save_pretrained(self.config["savepath"])


    def init_training(self):
        self.model.train()
    

    def to_device(self, device : str = None):

        if device is None:
            device = self.config["device"]

        self.model = self.model.to(device)


    def init_inference(self):
        self.model.eval()


    def load_model(self):
        
        if os.path.dirname(os.path.basename(self.config["huggingface_model"])) != "models":

            if self.config["access_token"] is not None:
                with open(self.config["access_token"], "r") as f:
                    access_token = f.read().strip()
            else:
                access_token = None

            low_cpu_mem_usage = True if self.config["load_in_8bit"] or self.config["load_in_4bit"] else False
            torch_dtype = torch.float16 if self.config["torch_dtype"] == "float16" else torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["huggingface_model"], 
                torch_dtype=torch_dtype,
                padding_side="left",
                token=access_token,
            )

            assert not self.config["load_in_4bit"] or not self.config["load_in_8bit"], "Must load model in 4-bit or 8-bit. Not both."

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config["load_in_4bit"],
                load_in_8bit=self.config["load_in_8bit"],
            )

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config["huggingface_model"],
                quantization_config=quantization_config,
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch_dtype,
                token=access_token,
            )
            
            if self.config["load_in_4bit"] or self.config["load_in_8bit"]:
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
                )

    
    def generate(self, prompts : List, generation_config : Dict) :
        
        fixed_outputs = []
        
        max_new_tokens = generation_config["max_new_tokens"]

        for doc in tqdm(prompts, desc="generating outputs", disable=False):
            
            inputs = self.tokenizer(
                return_tensors="pt", 
                padding="longest", 
                truncation=True,
                text=[doc],
                )            
        
            if "cuda" in str(self.model.device):
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_length=inputs["input_ids"].size(1) + max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    )

                outputs = outputs[:, inputs["input_ids"].size(1):]
        
            decoded_outputs = self.tokenizer.batch_decode(
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
                sequences=outputs, 
                )
        
            decoded_outputs = [self._parse_outputs(decoded_output) for decoded_output in decoded_outputs]

            fixed_outputs.extend(decoded_outputs)
            
            if "cuda" in str(self.model.device):
                torch.cuda.empty_cache()

        return fixed_outputs
            
            
    def _parse_outputs(self, decoded_output):
        
        parsed_outputs = repair_json(decoded_output, return_objects=True)
        
        if isinstance(parsed_outputs, list):
            parsed_outputs = parsed_outputs[-1]
        
        return parsed_outputs

    