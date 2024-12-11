from typing import List, Dict, Any
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader, Dataset

from huggingface_hub import login
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch


class LLMBaseTextGen:

    def __init__(
            self, 
            config : dict,
            local_path : str = None,
            ):
        

        self.model_config = config["model_config"]
        self.generation_config = config["generation_config"]

        if self.model_config["access_token"] != "none":
            with open(self.model_config["access_token"], "r") as f:
                self.access_token = f.read().strip()

            login(token=self.access_token)

        self.local_path = local_path


    def load_model(self):
        
        if self.local_path is None:

            low_cpu_mem_usage = True if self.model_config["load_in_8bit"] or self.model_config["load_in_4bit"] else False
            torch_dtype = torch.float16 if self.model_config["torch_dtype"] == "float16" else torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["huggingface_model"], 
                padding_side="left",
            )

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

            generation_config = GenerationConfig(
                max_new_tokens=30,
                do_sample=False,
            )

            self.model.generation_config = generation_config

        else:

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_path, 
                padding_side="left",
                )
            
            torch_dtype = torch.float16 if self.model_config["torch_dtype"] == "float16" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_path,
                )


class LLMBaseTextGenSeqCls(LLMBaseTextGen):

    def __init__(
            self,
            config : dict,
            n_labels : int,
            ):
        
        super().__init__(config)
        super().load_model()

        self.base_model = self.model
        self.n_labels = n_labels

        self.base_model.to(config["model_config"]["device"])

        self.base_model.config.output_hidden_states = True

        self.classifier_head = nn.Linear(self.base_model.config.hidden_size, self.n_labels).to(config["model_config"]["device"])
    

    def init_train(self):
        self.base_model.train()
    

    def init_eval(self):
        self.base_model.eval()


    def to(self, device):
        self.base_model.to(device)


    def forward(self, input_ids, attention_mask=None, labels=None):

        text_gen_output = self.base_model(input_ids, attention_mask)

        hidden_states = text_gen_output.hidden_states[-1]
                
        pooled_output = hidden_states[:, -1, :]  
        
        logits = self.classifier_head(pooled_output)  

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    

