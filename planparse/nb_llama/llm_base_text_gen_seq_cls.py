from typing import List, Dict, Any
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import numpy as np
import torch


class CausalTextClSModel(nn.Module):

    def __init__(
        self, 
        config, 
        local_path=None, 
        n_labels=None
        ):

        super().__init__()
        
        self.config = config
        self.local_path = local_path
        self.n_labels = n_labels
        self.base_model = None
        self.tokenizer = None
        self.classifier_head = None


    def load_base_model(self):
    
        low_cpu_mem_usage = (
            self.config["model_config"]["load_in_8bit"] 
            or self.config["model_config"]["load_in_4bit"]
        )
    
        torch_dtype = (
            torch.float16 if self.config["model_config"]["torch_dtype"] == "float16" else torch.float32
        )
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_config"]["huggingface_model"],
            padding_side="left",
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.config["model_config"]["load_in_4bit"],
            load_in_8bit=self.config["model_config"]["load_in_8bit"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.config["model_config"]["huggingface_model"],
            quantization_config=quantization_config,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
        )

        if self.config["model_config"]["load_in_4bit"] or self.config["model_config"]["load_in_8bit"]:
            model = prepare_model_for_kbit_training(model)

        model_config = LoraConfig(
            lora_dropout=self.config["model_config"]["lora_dropout"],
            lora_alpha=self.config["model_config"]["lora_alpha"],
            r=self.config["model_config"]["r"],
            task_type="CAUSAL_LM",
            bias="none",
        )

        self.base_model = get_peft_model(model, model_config)

        generation_config = GenerationConfig(
            max_new_tokens=30,
            do_sample=False,
        )
        self.base_model.generation_config = generation_config

        print(f"Loaded base model from {self.config['model_config']['huggingface_model']}")

        assert self.base_model is not None, "Base model failed to load."
        assert self.tokenizer is not None, "Tokenizer failed to load."
    
    
    def init_seq_cls_head(self):
        
        self.base_model._keys_to_ignore_on_load_missing = [
            "lm_head.weight",
            "lm_head.bias",
        ]

        for param in self.base_model.lm_head.parameters():
            param.requires_grad = False

        self.base_model.lm_head = None
        self.base_model.config.output_hidden_states = True
        self.classifier_head = nn.Linear(self.base_model.config.hidden_size, self.n_labels)

        print(f"Initialized classifier head with shape {self.classifier_head.weight.shape}")

        assert self.classifier_head is not None, "Classifier head initialization failed."
        
    

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
    

    def to(self, device):
        
        self.base_model.to(device)
        self.classifier_head.to(device)
        
        self.config["model_config"]["device"] = device

    def train(self):
        
        self.base_model.train()    
        self.classifier_head.train()

    def eval(self):
        
        self.base_model.eval()
        self.classifier_head.eval()


    # def __call__(self, input_ids, attention_mask=None, labels=None):
    #     return self.forward(input_ids, attention_mask, labels)


    def parameters(self):
        return self.base_model.parameters(recurse=True)


    def save_pretrained(self, save_directory):
        """
        Save the model, tokenizer, and additional components to the specified directory.
        """
        # Save the base model and tokenizer
        self.base_model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

        # Save classifier head if it exists
        if self.model_type == "seq_cls" and self.classifier_head:
            classifier_state = {
                "state_dict": self.classifier_head.state_dict(),
                "hidden_size": self.base_model.config.hidden_size,
                "n_labels": self.n_labels,
            }
            torch.save(classifier_state, f"{save_directory}/classifier_head.pt")

        # Save configuration
        with open(f"{save_directory}/model_config.json", "w") as f:
            json.dump(self.config, f)
    

    @classmethod
    def from_pretrained(cls, load_directory):
        """
        Load a model, tokenizer, and additional components from the specified directory.
        """
        # Load configuration
        with open(f"{load_directory}/model_config.json", "r") as f:
            config = json.load(f)
        
        # Determine model type
        classifier_head_path = f"{load_directory}/classifier_head.pt"
        n_labels = None
        if os.path.exists(classifier_head_path):
            classifier_state = torch.load(classifier_head_path)
            n_labels = classifier_state["n_labels"]
        
        # Initialize model
        instance = cls(config=config, local_path=load_directory, n_labels=n_labels)
        
        # Load classifier head if it exists
        if n_labels is not None:
            instance.classifier_head.load_state_dict(classifier_state["state_dict"])
        
        return instance
    
    